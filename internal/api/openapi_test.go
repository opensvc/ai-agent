package api

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/opensvc/ai-agent/internal/conversation"
)

type openAPIOperation struct {
	OperationID string                     `json:"operationId"`
	Responses   map[string]json.RawMessage `json:"responses"`
}

func loadOpenAPI(t *testing.T) (map[string]any, map[string]map[string]json.RawMessage) {
	t.Helper()
	var document map[string]any
	if err := json.Unmarshal(openAPIDocument, &document); err != nil {
		t.Fatalf("OpenAPI document is not JSON: %v", err)
	}
	var paths struct {
		Paths map[string]map[string]json.RawMessage `json:"paths"`
	}
	if err := json.Unmarshal(openAPIDocument, &paths); err != nil {
		t.Fatal(err)
	}
	return document, paths.Paths
}

func TestOpenAPIIsServedWithoutAuthentication(t *testing.T) {
	handler := newTestHandler(t, turnFunc(nil))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/openapi.json", nil))
	if response.Code != http.StatusOK || response.Header().Get("Content-Type") != "application/json" || !bytes.Equal(response.Body.Bytes(), openAPIDocument) {
		t.Fatalf("status=%d content type=%q", response.Code, response.Header().Get("Content-Type"))
	}
}

// TestOpenAPIDescribesTheRegisteredRoutes keeps the document equal to the
// routes of the API: a route added or removed without the document fails.
func TestOpenAPIDescribesTheRegisteredRoutes(t *testing.T) {
	_, paths := loadOpenAPI(t)
	var documented []string
	operationIDs := map[string]bool{}
	for path, item := range paths {
		for method, raw := range item {
			if method == "parameters" {
				continue
			}
			documented = append(documented, strings.ToUpper(method)+" "+path)
			var operation openAPIOperation
			if err := json.Unmarshal(raw, &operation); err != nil {
				t.Fatal(err)
			}
			if operation.OperationID == "" || operationIDs[operation.OperationID] || len(operation.Responses) == 0 {
				t.Fatalf("%s %s has a missing or duplicate operationId, or no response", method, path)
			}
			operationIDs[operation.OperationID] = true
		}
	}
	registered := []string{"GET /health", "GET /openapi.json"}
	for _, route := range backendRoutes("", nil, nil, nil, nil, auditLogger{}) {
		registered = append(registered, route.method+" /v1/{backend}"+route.path)
	}
	sort.Strings(documented)
	sort.Strings(registered)
	if !slices.Equal(documented, registered) {
		t.Fatalf("documented routes:\n%s\nregistered routes:\n%s", strings.Join(documented, "\n"), strings.Join(registered, "\n"))
	}
}

func TestOpenAPIReferencesResolve(t *testing.T) {
	document, _ := loadOpenAPI(t)
	components := document["components"].(map[string]any)
	var walk func(any)
	walk = func(node any) {
		switch value := node.(type) {
		case map[string]any:
			if target, ok := value["$ref"].(string); ok {
				parts := strings.Split(strings.TrimPrefix(target, "#/components/"), "/")
				group, _ := components[parts[0]].(map[string]any)
				if len(parts) != 2 || group == nil || group[parts[1]] == nil {
					t.Errorf("unresolved reference %s", target)
				}
			}
			for _, child := range value {
				walk(child)
			}
		case []any:
			for _, child := range value {
				walk(child)
			}
		}
	}
	walk(document)
}

func TestOpenAPIDocumentsTheConversationErrorCodes(t *testing.T) {
	for _, err := range []error{
		conversation.ErrConfirmationExpired, conversation.ErrNotPending, conversation.ErrMessageTooLarge, conversation.ErrExpired,
		conversation.ErrNotFound, conversation.ErrBusy, conversation.ErrUserLimit, conversation.ErrLimit, conversation.ErrInvalid, errors.New("unexpected"),
	} {
		_, code, _ := conversationError(err)
		if !bytes.Contains(openAPIDocument, []byte("`"+code+"`")) {
			t.Errorf("error code %s is not documented", code)
		}
	}
	for _, code := range []string{"unauthorized", "authentication_unavailable", "too_many_requests", "unsupported_media_type", "request_too_large",
		"invalid_prompt", "prompt_too_large", "invalid_decision", "streaming_unavailable"} {
		if !bytes.Contains(openAPIDocument, []byte("`"+code+"`")) {
			t.Errorf("error code %s is not documented", code)
		}
	}
}
