package om3

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
)

// statusSession answers get_cluster_status with the versions of its nodes,
// pages the pages of them when set, and counts the calls.
type statusSession struct {
	versions  []string
	pages     [][]string
	noCursor  bool
	calls     int
	fail      bool
	toolError bool
	notListed bool
}

func (s *statusSession) ServerInfo() (string, string) { return "om3-mcp", "test" }
func (s *statusSession) ListTools(context.Context) ([]*mcp.Tool, error) {
	return nil, nil
}
func (s *statusSession) ListResources(context.Context) ([]*mcp.Resource, error) { return nil, nil }
func (s *statusSession) ReadResource(context.Context, string) (*mcp.ReadResourceResult, error) {
	return nil, errors.New("none")
}
func (s *statusSession) Close() error { return nil }
func (s *statusSession) CallTool(_ context.Context, name string, arguments map[string]any) (*mcp.CallToolResult, error) {
	s.calls++
	if s.notListed {
		return nil, fmt.Errorf("%w: %s", agent.ErrToolNotListed, name)
	}
	if s.fail || name != "get_cluster_status" {
		return nil, errors.New("unavailable")
	}
	if s.toolError {
		return &mcp.CallToolResult{IsError: true}, nil
	}
	pages := s.pages
	if pages == nil {
		pages = [][]string{s.versions}
	}
	index := 0
	if cursor, _ := arguments["node_cursor"].(string); cursor != "" {
		fmt.Sscanf(cursor, "page%d", &index)
	}
	var items []map[string]any
	for _, version := range pages[index] {
		items = append(items, map[string]any{"name": "n", "status": map[string]any{"agent_version": version}})
	}
	nodes := map[string]any{"items": items, "truncated": index+1 < len(pages)}
	if index+1 < len(pages) && !s.noCursor {
		nodes["next_cursor"] = fmt.Sprintf("page%d", index+1)
	}
	return &mcp.CallToolResult{StructuredContent: map[string]any{"nodes": nodes}}, nil
}

func clusterContext(cluster string) context.Context {
	headers := http.Header{}
	headers.Set(ClusterIDHeader, cluster)
	return auth.WithMCPHeaders(context.Background(), headers)
}

// TestClusterVersionIsTheNewestOfItsNodesReadOnce reads the newest version
// the nodes report: a conversation of om3 is about the cluster, which runs
// its newest version once it is upgraded.
func TestClusterVersionIsTheNewestOfItsNodesReadOnce(t *testing.T) {
	versions := NewClusterVersions()
	clock := time.Now()
	versions.now = func() time.Time { return clock }
	session := &statusSession{versions: []string{"v3.1.0-rc2", "v3.0.0-rc56-41-g451250d44", ""}}
	for range 3 {
		if got, _ := versions.Version(clusterContext("c1"), session); got != "v3.1.0-rc2" {
			t.Fatalf("version = %q", got)
		}
	}
	if session.calls != 1 {
		t.Errorf("MCP calls = %d, want 1 kept for the cluster", session.calls)
	}
	// Another cluster is asked on its own, and an upgrade is seen once the
	// version kept is too old.
	versions.Version(clusterContext("c2"), session)
	if session.calls != 2 {
		t.Errorf("MCP calls = %d, want 2", session.calls)
	}
	session.versions = []string{"v3.1.0"}
	clock = clock.Add(versionLifetime)
	if got, _ := versions.Version(clusterContext("c1"), session); got != "v3.1.0" || session.calls != 3 {
		t.Errorf("version past its lifetime = %q after %d calls", got, session.calls)
	}
}

// TestClusterVersionIsEmptyWhenTheMCPGivesNone gives no version, and no
// error, for a tool error or nodes reporting none: the search takes the
// latest book. A failed transport is an error, which ends the turn.
func TestClusterVersionIsEmptyWhenTheMCPGivesNone(t *testing.T) {
	versions := NewClusterVersions()
	for _, session := range []*statusSession{{toolError: true}, {versions: []string{"", "unknown"}}} {
		if got, err := versions.Version(clusterContext("c1"), session); got != "" || err != nil {
			t.Errorf("version = %q, %v", got, err)
		}
	}
	if got, err := versions.Version(clusterContext("c1"), &statusSession{fail: true}); got != "" || err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Errorf("a failed transport: version = %q, %v", got, err)
	}
	// An MCP that does not list the tool, the agent refusing the call, gives
	// no version and no error: the search takes the latest book.
	if got, err := versions.Version(clusterContext("c1"), &statusSession{notListed: true}); got != "" || err != nil {
		t.Errorf("a tool not listed: version = %q, %v", got, err)
	}
}

func TestClusterVersionReadsEveryPageOfNodes(t *testing.T) {
	// The newest node is on the last page.
	session := &statusSession{pages: [][]string{{"v3.1.0"}, {"v3.0.0-rc56"}, {"v3.1.2"}}}
	if got, _ := NewClusterVersions().Version(clusterContext("c1"), session); got != "v3.1.2" || session.calls != 3 {
		t.Errorf("version = %q after %d calls", got, session.calls)
	}
	// A cluster of more pages than read, or a page truncated with no
	// cursor, has a version not known rather than that of part of it.
	many := make([][]string, versionPages+1)
	for i := range many {
		many[i] = []string{"v3.1.0"}
	}
	session = &statusSession{pages: many}
	if got, _ := NewClusterVersions().Version(clusterContext("c1"), session); got != "" || session.calls != versionPages {
		t.Errorf("too many pages: version = %q after %d calls", got, session.calls)
	}
	if got, _ := NewClusterVersions().Version(clusterContext("c1"), &statusSession{pages: [][]string{{"v3.1.0"}, {"v3.0.0"}}, noCursor: true}); got != "" {
		t.Errorf("no cursor: version = %q", got)
	}
}

// TestClusterVersionIsTheNewestCommitInARollingUpgrade reads, of nodes of one
// release line in a rolling upgrade, the newest commit, whatever their order.
func TestClusterVersionIsTheNewestCommitInARollingUpgrade(t *testing.T) {
	for _, versions := range [][]string{
		{"v3.0.0-rc47-2-gaaaaaaa", "v3.0.0-rc46-12-gbbbbbbb"},
		{"v3.0.0-rc46-12-gbbbbbbb", "v3.0.0-rc47-2-gaaaaaaa"},
	} {
		if got, _ := NewClusterVersions().Version(clusterContext("c1"), &statusSession{versions: versions}); got != "v3.0.0-rc47-2-gaaaaaaa" {
			t.Errorf("%v: version = %q", versions, got)
		}
	}
}
