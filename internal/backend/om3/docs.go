package om3

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/auth"
	"github.com/opensvc/ai-agent/internal/docs"
)

// versionLifetime is how long the version the MCP gave for a cluster is
// kept: an upgrade is seen within it.
const versionLifetime = 5 * time.Minute

const (
	// DocsDefault says which book a search gives when the model names no
	// release line.
	DocsDefault = "the release line the nodes of this cluster run, the oldest when they differ"
	// DocsUnknown says the version of the cluster could not be read.
	DocsUnknown = "The version the nodes of this cluster run could not be read"
)

// ClusterVersions reads the version the nodes of the cluster of a request
// run, through the MCP, for the documentation to give the book of its
// release line.
type ClusterVersions struct {
	now func() time.Time

	mu       sync.Mutex
	versions map[string]knownVersion
}

// knownVersion is the version the MCP gave for a cluster, and when.
type knownVersion struct {
	version string
	at      time.Time
}

// NewClusterVersions makes an empty cache of cluster versions.
func NewClusterVersions() *ClusterVersions {
	return &ClusterVersions{now: time.Now, versions: map[string]knownVersion{}}
}

// Version is the oldest version the nodes of the cluster of the request
// report, read through the MCP and kept versionLifetime; empty when the MCP
// gives none, such as a tool error. A failed MCP transport is an error,
// which ends the turn.
func (d *ClusterVersions) Version(ctx context.Context, session agent.MCPSession) (string, error) {
	cluster := auth.MCPHeadersFromContext(ctx).Get(ClusterIDHeader)
	d.mu.Lock()
	known, ok := d.versions[cluster]
	d.mu.Unlock()
	if ok && d.now().Sub(known.at) < versionLifetime {
		return known.version, nil
	}
	version, err := reportedVersion(ctx, session)
	if err != nil {
		return "", err
	}
	if version != "" && cluster != "" {
		d.mu.Lock()
		d.versions[cluster] = knownVersion{version: version, at: d.now()}
		d.mu.Unlock()
	}
	return version, nil
}

const (
	// versionPageNodes and versionPages bound the read of the versions of
	// the nodes: pages of 200 nodes, 5 at most, a larger cluster read as
	// one whose version is not known rather than by part of its nodes.
	versionPageNodes = 200
	versionPages     = 5
)

// nodePage is the page of nodes of a cluster status: their agent versions,
// and the cursor of the next page.
type nodePage struct {
	Nodes struct {
		Items []struct {
			Status *struct {
				AgentVersion string `json:"agent_version"`
			} `json:"status"`
		} `json:"items"`
		NextCursor string `json:"next_cursor"`
		Truncated  bool   `json:"truncated"`
	} `json:"nodes"`
}

// reportedVersion is the oldest version the nodes of the cluster report, the
// commits past a tag counted, every page of them read; empty when a page is a tool error
// or cannot be read, the nodes are more than versionPages pages, or none
// reports a version; an error when the MCP transport fails.
func reportedVersion(ctx context.Context, session agent.MCPSession) (string, error) {
	oldest, cursor := "", ""
	for range versionPages {
		arguments := map[string]any{"node_limit": versionPageNodes, "object_limit": 1}
		if cursor != "" {
			arguments["node_cursor"] = cursor
		}
		result, err := session.CallTool(ctx, "get_cluster_status", arguments)
		// An MCP that does not list the tool gives no version: the search
		// takes the latest book.
		if errors.Is(err, agent.ErrToolNotListed) {
			return "", nil
		}
		if err != nil {
			return "", fmt.Errorf("read the version of the cluster: %w", err)
		}
		if result == nil || result.IsError || result.StructuredContent == nil {
			return "", nil
		}
		data, err := json.Marshal(result.StructuredContent)
		var page nodePage
		if err != nil || json.Unmarshal(data, &page) != nil {
			return "", nil
		}
		for _, node := range page.Nodes.Items {
			if node.Status == nil || docs.Line(node.Status.AgentVersion) == "" {
				continue
			}
			if oldest == "" || docs.OlderVersion(node.Status.AgentVersion, oldest) {
				oldest = node.Status.AgentVersion
			}
		}
		if !page.Nodes.Truncated {
			return oldest, nil
		}
		if cursor = page.Nodes.NextCursor; cursor == "" {
			return "", nil
		}
	}
	return "", nil
}
