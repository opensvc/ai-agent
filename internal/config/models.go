package config

import (
	"fmt"
	"os"
	"regexp"
	"slices"
	"strings"

	"github.com/opensvc/ai-agent/internal/llm"
)

const (
	// ModelsEnv lists the models the operator offers besides the default.
	ModelsEnv = "OPENSVC_AI_MODELS"
	// modelPrefix starts the variables of a model of ModelsEnv: the model
	// "haiku" reads OPENSVC_AI_MODEL_HAIKU_PROTOCOL and the like.
	modelPrefix = "OPENSVC_AI_MODEL_"
	// LLMLabelEnv names the default model in the interfaces.
	LLMLabelEnv = llmPrefix + "LABEL"
)

// modelIDPattern is the form of the ID of a model of ModelsEnv: lowercase,
// with neither dash nor underscore, so that in capitals it names its
// variables unambiguously.
var modelIDPattern = regexp.MustCompile(`^[a-z][a-z0-9]{0,15}$`)

// modelKeys are the variables of a model, after its prefix.
var modelKeys = []string{
	"PROTOCOL", "BASE_URL", "MODEL", "LABEL", "AUTH_MODE", "AUTH_HEADER", "API_TOKEN", "API_TOKEN_FROM",
	"TIMEOUT", "MAX_OUTPUT_TOKENS", "REASONING_EFFORT", "STORE",
}

// ModelConfig is a model the agent offers: its ID, its label and its LLM
// configuration.
type ModelConfig struct {
	ID    string
	Label string
	LLM   LLMConfig
}

// LoadModels loads the models of the agent: the default of the
// OPENSVC_AI_LLM_* variables, first, then those of OPENSVC_AI_MODELS, in
// their order.
func LoadModels() ([]ModelConfig, error) {
	return loadModels(os.Getenv, os.Environ())
}

func loadModels(getenv func(string) string, environ []string) ([]ModelConfig, error) {
	// The default has its own key: it shares none of a model of
	// OPENSVC_AI_MODELS, which may share its key instead.
	if getenv(llmPrefix+"API_TOKEN_FROM") != "" {
		return nil, fmt.Errorf("%sAPI_TOKEN_FROM is not read: the default model takes its key in %s; a model of %s takes the key of the default with API_TOKEN_FROM=default", llmPrefix, LLMAPITokenEnv, ModelsEnv)
	}
	defaultLLM, err := loadLLM(getenv)
	if err != nil {
		return nil, err
	}
	defaultLabel, err := modelLabel(LLMLabelEnv, defaultLLM.Model, getenv)
	if err != nil {
		return nil, err
	}
	models := []ModelConfig{{ID: llm.DefaultModelID, Label: defaultLabel, LLM: defaultLLM}}

	ids, err := modelIDs(getenv(ModelsEnv))
	if err != nil {
		return nil, err
	}
	if err := checkModelVariables(ids, environ); err != nil {
		return nil, err
	}
	for _, id := range ids {
		prefix := modelPrefix + strings.ToUpper(id) + "_"
		tokenEnv, err := modelTokenEnv(prefix, ids, getenv)
		if err != nil {
			return nil, err
		}
		config, err := loadLLMVariables(prefix, tokenEnv, getenv)
		if err != nil {
			return nil, err
		}
		label, err := modelLabel(prefix+"LABEL", config.Model, getenv)
		if err != nil {
			return nil, err
		}
		models = append(models, ModelConfig{ID: id, Label: label, LLM: config})
	}
	return models, nil
}

// modelIDs reads the IDs of ModelsEnv: at most llm.MaxModels-1, the default
// taking the first place, each of modelIDPattern, neither repeated nor the
// default.
func modelIDs(value string) ([]string, error) {
	if strings.TrimSpace(value) == "" {
		return nil, nil
	}
	var ids []string
	for _, id := range strings.Split(value, ",") {
		id = strings.TrimSpace(id)
		switch {
		case id == llm.DefaultModelID:
			return nil, fmt.Errorf("%s: %q names the model of the OPENSVC_AI_LLM_* variables", ModelsEnv, id)
		case !modelIDPattern.MatchString(id):
			return nil, fmt.Errorf("%s: model ID %q must be a lowercase letter then up to 15 lowercase letters or digits", ModelsEnv, id)
		case slices.Contains(ids, id):
			return nil, fmt.Errorf("%s: model ID %q is repeated", ModelsEnv, id)
		}
		ids = append(ids, id)
	}
	if len(ids) > llm.MaxModels-1 {
		return nil, fmt.Errorf("%s: at most %d models besides the default", ModelsEnv, llm.MaxModels-1)
	}
	return ids, nil
}

// checkModelVariables refuses a variable of a model that ModelsEnv does not
// list, or that no model reads: a typo would otherwise go unnoticed.
func checkModelVariables(ids []string, environ []string) error {
	// In the order of their names, so that the error is the same at each
	// start.
	names := make([]string, 0, len(environ))
	for _, entry := range environ {
		name, _, _ := strings.Cut(entry, "=")
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		rest, ok := strings.CutPrefix(name, modelPrefix)
		if !ok {
			continue
		}
		id, key, _ := strings.Cut(rest, "_")
		if !slices.Contains(ids, strings.ToLower(id)) || id != strings.ToUpper(id) {
			return fmt.Errorf("%s: model %q is not in %s: list it there, or remove its variables", name, strings.ToLower(id), ModelsEnv)
		}
		if !slices.Contains(modelKeys, key) {
			return fmt.Errorf("%s: unknown variable of model %q", name, strings.ToLower(id))
		}
	}
	return nil
}

// modelTokenEnv is the variable of the key of the model of prefix: its own
// API_TOKEN, or the one of the model API_TOKEN_FROM names, which has its
// own. The agent keeps the name of the variable, never the key.
func modelTokenEnv(prefix string, ids []string, getenv func(string) string) (string, error) {
	from := strings.TrimSpace(getenv(prefix + "API_TOKEN_FROM"))
	if from == "" {
		return prefix + "API_TOKEN", nil
	}
	if getenv(prefix+"API_TOKEN") != "" {
		return "", fmt.Errorf("%sAPI_TOKEN and %sAPI_TOKEN_FROM are exclusive", prefix, prefix)
	}
	if strings.TrimSpace(getenv(prefix+"AUTH_MODE")) == LLMAuthModeNone {
		return "", fmt.Errorf("%sAPI_TOKEN_FROM requires an authentication", prefix)
	}
	if from == llm.DefaultModelID {
		return LLMAPITokenEnv, nil
	}
	if !slices.Contains(ids, from) {
		return "", fmt.Errorf("%sAPI_TOKEN_FROM %q is not a model", prefix, from)
	}
	source := modelPrefix + strings.ToUpper(from) + "_"
	if source == prefix || strings.TrimSpace(getenv(source+"API_TOKEN_FROM")) != "" {
		return "", fmt.Errorf("%sAPI_TOKEN_FROM %q must name a model with its own API_TOKEN", prefix, from)
	}
	return source + "API_TOKEN", nil
}

// modelLabel reads the label of a model, its name at the provider when the
// variable is empty, cut to llm.MaxModelLabelRunes characters: a long name,
// such as the ARN of an inference profile, starts as it did before labels.
func modelLabel(variable, model string, getenv func(string) string) (string, error) {
	label := strings.TrimSpace(getenv(variable))
	if label == "" {
		label = model
		if runes := []rune(label); len(runes) > llm.MaxModelLabelRunes {
			label = strings.TrimSpace(string(runes[:llm.MaxModelLabelRunes]))
		}
	}
	if err := llm.ValidateModelLabel(label); err != nil {
		return "", fmt.Errorf("%s: %w", variable, err)
	}
	return label, nil
}
