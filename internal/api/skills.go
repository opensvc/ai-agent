package api

import (
	"context"
	"log"
	"log/slog"
	"net/http"

	"github.com/opensvc/ai-agent/internal/agent"
	"github.com/opensvc/ai-agent/internal/requestid"
)

// SkillLister lists the skills the MCP of a backend serves to the user of ctx.
type SkillLister interface {
	Skills(context.Context) ([]agent.SkillInfo, error)
}

type SkillResponse struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type SkillListResponse struct {
	Skills []SkillResponse `json:"skills"`
}

// serveSkills lists the skills a run may ask in forwardedProps.skill: those
// the MCP serves to the user, as the collector publishes them to their
// groups. None when the MCP serves no skill.
func serveSkills(skills SkillLister, audit auditLogger) http.HandlerFunc {
	return func(response http.ResponseWriter, request *http.Request) {
		if skills == nil {
			writeJSONError(response, http.StatusNotFound, "not_found", "this backend lists no skills")
			return
		}
		items, err := skills.Skills(request.Context())
		if err != nil {
			// The cause in the server log; the audit keeps the code only.
			log.Printf("list the skills of the MCP, request %s: %v", requestid.FromContext(request.Context()), err)
			audit.event(request.Context(), "skills_list_failed", slog.Int("status", http.StatusBadGateway), slog.String("code", "skills_unavailable"))
			writeJSONError(response, http.StatusBadGateway, "skills_unavailable", "the skills of the MCP could not be read")
			return
		}
		result := make([]SkillResponse, 0, len(items))
		for _, item := range items {
			result = append(result, SkillResponse{Name: item.Name, Description: item.Description})
		}
		audit.event(request.Context(), "skills_listed", slog.Int("count", len(result)))
		response.Header().Set("Cache-Control", "no-store")
		writeJSON(response, http.StatusOK, SkillListResponse{Skills: result})
	}
}
