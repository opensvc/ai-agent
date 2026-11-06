package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// The _meta keys of the MCP tools the confirmation of a call reads: a tool
// hidden is never offered to the model; a tool confirmed names the tool that
// tells what its call would do, which takes the arguments of the call, or
// {tool, arguments} when it wraps them.
const (
	hiddenToolMeta   = "opensvc/hidden"
	previewToolMeta  = "opensvc/preview_tool"
	previewWrapsMeta = "opensvc/preview_wraps_arguments"
)

const (
	// previewTimeout bounds the call of a preview: a confirmation never
	// waits long on it.
	previewTimeout = 5 * time.Second
	// maxPreviewBytes bounds a preview a confirmation carries.
	maxPreviewBytes = 256 << 10
	// maxPreviewErrorBytes bounds the cause of a preview that failed.
	maxPreviewErrorBytes = 1 << 10
)

// hiddenTool tells whether a tool is for the client only, such as a preview.
func hiddenTool(tool *mcp.Tool) bool {
	return tool != nil && tool.Meta[hiddenToolMeta] == true
}

// modelTools are the tools the model may be offered: the hidden ones and
// the previews the others name are left out.
func modelTools(tools []*mcp.Tool) []*mcp.Tool {
	previews := map[string]bool{}
	for _, tool := range tools {
		if name, ok := tool.Meta[previewToolMeta].(string); ok {
			previews[name] = true
		}
	}
	return slices.DeleteFunc(slices.Clone(tools), func(tool *mcp.Tool) bool {
		return hiddenTool(tool) || previews[tool.Name]
	})
}

// preview tells what a call of tool with arguments would do, by the preview
// tool it names, called as the user: the structured content of its result,
// or the cause of its failure. A tool naming none has no preview, and a
// failure blocks nothing: the confirmation shows the arguments.
func preview(ctx context.Context, session MCPSession, tools map[string]*mcp.Tool, tool *mcp.Tool, arguments json.RawMessage) (json.RawMessage, string) {
	name, ok := tool.Meta[previewToolMeta].(string)
	if !ok || name == "" {
		return nil, ""
	}
	// A tool listed without annotations is not read-only.
	if target, ok := tools[name]; !ok || target.Annotations == nil || !target.Annotations.ReadOnlyHint {
		return nil, fmt.Sprintf("the preview tool %s is not a read-only tool of the server", name)
	}
	var input map[string]any
	if err := json.Unmarshal(arguments, &input); err != nil || input == nil {
		input = map[string]any{}
	}
	if tool.Meta[previewWrapsMeta] == true {
		input = map[string]any{"tool": tool.Name, "arguments": input}
	}
	ctx, cancel := context.WithTimeout(ctx, previewTimeout)
	defer cancel()
	result, err := session.CallTool(ctx, name, input)
	switch {
	case err != nil:
		return nil, boundPreviewError("the preview failed: " + err.Error())
	case result == nil:
		return nil, "the preview returned no result"
	case result.IsError:
		var texts []string
		for _, content := range result.Content {
			if text, ok := content.(*mcp.TextContent); ok {
				texts = append(texts, text.Text)
			}
		}
		return nil, boundPreviewError("the preview was refused: " + strings.Join(texts, " "))
	case result.StructuredContent == nil:
		return nil, "the preview returned no structured content"
	}
	data, err := json.Marshal(result.StructuredContent)
	if err != nil {
		return nil, "the preview could not be encoded"
	}
	if len(data) > maxPreviewBytes {
		return nil, fmt.Sprintf("the preview exceeds %d bytes", maxPreviewBytes)
	}
	return data, ""
}

// boundPreviewError cuts the cause of a failed preview, on a rune boundary.
func boundPreviewError(text string) string {
	if len(text) <= maxPreviewErrorBytes {
		return text
	}
	return strings.ToValidUTF8(text[:maxPreviewErrorBytes], "") + "…"
}
