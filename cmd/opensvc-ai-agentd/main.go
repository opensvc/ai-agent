package main

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/api"
	"github.com/opensvc/ai-agent/internal/backend/oc3"
	"github.com/opensvc/ai-agent/internal/backend/om3"
	"github.com/opensvc/ai-agent/internal/buildinfo"
	"github.com/opensvc/ai-agent/internal/config"
	"github.com/opensvc/ai-agent/internal/conversation"
	conversationsqlite "github.com/opensvc/ai-agent/internal/conversation/sqlite"
	"github.com/opensvc/ai-agent/internal/docs"
	"github.com/opensvc/ai-agent/internal/llm"
	"github.com/opensvc/ai-agent/internal/llmfactory"
	"github.com/opensvc/ai-agent/internal/mcpclient"
)

const maxHTTPHeaderBytes = 64 << 10

func main() {
	if len(os.Args) > 1 && os.Args[1] == "feedback" {
		if err := runFeedback(os.Args[2:], os.Stdout); err != nil {
			fmt.Fprintf(os.Stderr, "opensvc-ai-agentd feedback: %v\n", err)
			os.Exit(2)
		}
		return
	}
	processConfig, err := config.Load()
	if err != nil {
		log.Fatalf("load configuration: %v", err)
	}
	llmConfig, err := config.LoadLLM()
	if err != nil {
		log.Fatalf("load LLM configuration: %v", err)
	}
	agentConfig, err := config.LoadAgent()
	if err != nil {
		log.Fatalf("load agent configuration: %v", err)
	}
	conversationConfig, err := config.LoadConversation()
	if err != nil {
		log.Fatalf("load conversation configuration: %v", err)
	}
	mcpConfig, err := config.LoadMCP()
	if err != nil {
		log.Fatalf("load MCP configuration: %v", err)
	}
	docsConfig, err := config.LoadDocs()
	if err != nil {
		log.Fatalf("load documentation configuration: %v", err)
	}

	model, err := llmfactory.New(llmConfig, nil)
	if err != nil {
		log.Fatalf("create LLM client: %v", err)
	}
	var om3Docs *docs.Library
	om3Prompt := om3.SystemPrompt
	if library := loadDocs(docsConfig.URL); library != nil {
		om3Docs = library
		om3Prompt += "\n" + om3.DocsPrompt
	}
	// om3 is always served; oc3 only when its MCP socket is configured.
	specs := []backendSpec{{
		name: "om3", socketPath: mcpConfig.OM3SocketPath, systemPrompt: om3Prompt, docs: om3Docs, newMCPClient: om3.NewMCPClient,
		newAuthenticator: func(client *mcpclient.Client) api.Authenticator { return om3.NewAuthenticator(client) },
	}}
	if mcpConfig.OC3SocketPath != "" {
		specs = append(specs, backendSpec{
			name: "oc3", socketPath: mcpConfig.OC3SocketPath, systemPrompt: oc3.SystemPrompt, newMCPClient: oc3.NewMCPClient,
			newAuthenticator: func(client *mcpclient.Client) api.Authenticator { return oc3.NewAuthenticator(client) },
		})
	}
	startupContext, cancelStartup := context.WithTimeout(context.Background(), 30*time.Second)
	conversationStore, err := conversationsqlite.Open(startupContext, conversationsqlite.Config{
		Path:                 conversationConfig.DatabasePath,
		MaxConversations:     conversationConfig.MaxConversations,
		MaxUserConversations: conversationConfig.MaxUserConversations,
		MaxDatabaseBytes:     conversationConfig.MaxDatabaseBytes,
	})
	if err != nil {
		cancelStartup()
		log.Fatalf("open conversation store: %v", err)
	}
	var backends []api.Backend
	var conversations []*conversation.Service
	for _, spec := range specs {
		serviceConfig := conversation.ServiceConfig{
			Backend: spec.name, Lifetime: conversationConfig.Lifetime, FeedbackRetention: conversationConfig.FeedbackRetention,
			Agent: conversation.NewAgentInfo(buildinfo.Version(), llmConfig.Model),
		}
		backend, service, err := newBackend(spec, model, conversationStore, agent.Config{
			Backend: spec.name, SystemPrompt: spec.systemPrompt, MaxIterations: agentConfig.MaxIterations, Timeout: agentConfig.Timeout,
			SearchThreshold: agentConfig.SearchThreshold, SearchScoreRatio: agentConfig.SearchScoreRatio, Docs: spec.docs,
		}, serviceConfig)
		if err != nil {
			cancelStartup()
			_ = conversationStore.Close()
			log.Fatalf("create %s backend: %v", spec.name, err)
		}
		backends = append(backends, backend)
		conversations = append(conversations, service)
		log.Printf("serving the %s backend under /v1/%s", spec.name, spec.name)
	}
	// Recovery and expiry apply to the whole store, every backend included.
	if recovered, recoverErr := conversations[0].Recover(startupContext); recoverErr != nil {
		cancelStartup()
		_ = conversationStore.Close()
		log.Fatalf("recover interrupted conversation turns: %v", recoverErr)
	} else if recovered != 0 {
		log.Printf("recovered %d interrupted conversation turns", recovered)
	}
	if _, err := conversations[0].DeleteExpired(startupContext); err != nil {
		cancelStartup()
		_ = conversationStore.Close()
		log.Fatalf("delete expired conversations: %v", err)
	}
	cancelStartup()
	defer func() {
		if err := conversationStore.Close(); err != nil {
			log.Printf("close conversation store: %v", err)
		}
	}()
	auditLogger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	handler, err := api.NewHandler(backends, api.HandlerConfig{
		MaxConcurrentAsks:  processConfig.MaxConcurrentAsks,
		AuditLogger:        auditLogger,
		CORSAllowedOrigins: processConfig.CORSAllowedOrigins,
	})
	if err != nil {
		_ = conversationStore.Close()
		log.Fatalf("create HTTP API: %v", err)
	}

	listener, tlsConfig, err := listenHTTPS(processConfig)
	if err != nil {
		_ = conversationStore.Close()
		log.Fatalf("listen for HTTP API: %v", err)
	}
	defer listener.Close()
	server := newHTTPServer(listener.Addr().String(), handler)
	server.TLSConfig = tlsConfig
	serveErrors := make(chan error, 1)
	go func() {
		serveErrors <- server.ServeTLS(listener, "", "")
	}()

	signalContext, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()
	log.Printf("opensvc-ai-agentd listening on https://%s", listener.Addr())
	select {
	case err := <-serveErrors:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("serve HTTP API: %v", err)
			return
		}
	case <-signalContext.Done():
		stopSignals()
		log.Printf("opensvc-ai-agentd shutting down with a %s deadline", processConfig.ShutdownTimeout)
		if err := shutdownHTTPServer(server, processConfig.ShutdownTimeout); err != nil {
			log.Printf("force HTTP API shutdown: %v", err)
		}
		if err := <-serveErrors; err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("serve HTTP API during shutdown: %v", err)
		}
	}
}

// backendSpec describes one backend: its MCP socket and the rules of its
// package.
type backendSpec struct {
	name         string
	socketPath   string
	systemPrompt string
	// docs is the book the model searches; nil for none.
	docs             *docs.Library
	newMCPClient     func(string) (*mcpclient.Client, error)
	newAuthenticator func(*mcpclient.Client) api.Authenticator
}

// loadDocs reads the book of the documentation at root, nil when none is
// configured or it cannot be read: the backend then offers no documentation
// search, and serves the rest. A root naming its llms.txt is its directory.
func loadDocs(root string) *docs.Library {
	if root == "" {
		log.Printf("documentation: none configured, no documentation search")
		return nil
	}
	index := strings.TrimSuffix(root, "llms.txt")
	if !strings.HasSuffix(index, "/") {
		index += "/"
	}
	index += "llms.txt"
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	library, _, err := docs.BookLoader(&http.Client{Timeout: 30 * time.Second})(ctx, index)
	if err != nil {
		log.Printf("documentation: read %s: %v; no documentation search", index, err)
		return nil
	}
	log.Printf("documentation: %s, %d sections, %d parents", library.Source(), library.Len(), library.Parents())
	return library
}

// newBackend builds the MCP client, the agent and the conversation service of
// one backend on the shared model and store.
func newBackend(spec backendSpec, model llm.Client, store conversation.Store, agentConfig agent.Config, serviceConfig conversation.ServiceConfig) (api.Backend, *conversation.Service, error) {
	client, err := spec.newMCPClient(spec.socketPath)
	if err != nil {
		return api.Backend{}, nil, fmt.Errorf("create MCP client: %w", err)
	}
	orchestrator, err := agent.New(model, func(ctx context.Context) (agent.MCPSession, error) {
		return client.Connect(ctx)
	}, agentConfig)
	if err != nil {
		return api.Backend{}, nil, fmt.Errorf("create agent: %w", err)
	}
	service, err := conversation.NewService(store, orchestrator, serviceConfig)
	if err != nil {
		return api.Backend{}, nil, fmt.Errorf("create conversation service: %w", err)
	}
	return api.Backend{Name: spec.name, Conversations: service, Authenticator: spec.newAuthenticator(client), Tools: orchestrator}, service, nil
}

// listenHTTPS loads the TLS identity before opening the TCP listener.
func listenHTTPS(cfg config.Config) (net.Listener, *tls.Config, error) {
	certificate, err := tls.LoadX509KeyPair(cfg.TLSCertFile, cfg.TLSKeyFile)
	if err != nil {
		return nil, nil, fmt.Errorf("load API TLS certificate/key: %w", err)
	}
	listener, err := net.Listen("tcp", cfg.ListenAddress)
	if err != nil {
		return nil, nil, fmt.Errorf("listen on TCP address %s: %w", cfg.ListenAddress, err)
	}
	return listener, &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{certificate}}, nil
}

func newHTTPServer(address string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              address,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		IdleTimeout:       2 * time.Minute,
		MaxHeaderBytes:    maxHTTPHeaderBytes,
	}
}

func shutdownHTTPServer(server *http.Server, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		shutdownErr := fmt.Errorf("graceful HTTP shutdown: %w", err)
		if closeErr := server.Close(); closeErr != nil && !errors.Is(closeErr, http.ErrServerClosed) {
			return errors.Join(shutdownErr, fmt.Errorf("close HTTP server: %w", closeErr))
		}
		return shutdownErr
	}
	return nil
}
