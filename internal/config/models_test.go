package config

import (
	"reflect"
	"strings"
	"testing"
)

// modelEnv is an environment of a default model and the models it adds.
func modelEnv(extra map[string]string) (func(string) string, []string) {
	values := map[string]string{
		"OPENSVC_AI_LLM_PROTOCOL":  "responses",
		"OPENSVC_AI_LLM_BASE_URL":  "https://api.openai.example.test/v1",
		"OPENSVC_AI_LLM_MODEL":     "gpt-luna",
		"OPENSVC_AI_LLM_AUTH_MODE": "bearer",
		LLMAPITokenEnv:             "default-key",
	}
	for key, value := range extra {
		if value == "" {
			delete(values, key)
			continue
		}
		values[key] = value
	}
	environ := make([]string, 0, len(values))
	for key, value := range values {
		environ = append(environ, key+"="+value)
	}
	return func(key string) string { return values[key] }, environ
}

// haikuEnv declares the model haiku, on the Messages protocol.
var haikuEnv = map[string]string{
	"OPENSVC_AI_MODEL_HAIKU_LABEL":     "Claude Haiku",
	"OPENSVC_AI_MODEL_HAIKU_PROTOCOL":  "messages",
	"OPENSVC_AI_MODEL_HAIKU_BASE_URL":  "https://api.anthropic.example.test/v1",
	"OPENSVC_AI_MODEL_HAIKU_MODEL":     "claude-haiku",
	"OPENSVC_AI_MODEL_HAIKU_AUTH_MODE": "api_key",
	"OPENSVC_AI_MODEL_HAIKU_API_TOKEN": "haiku-key",
}

func withEnv(base map[string]string, extra map[string]string) map[string]string {
	values := make(map[string]string, len(base)+len(extra))
	for key, value := range base {
		values[key] = value
	}
	for key, value := range extra {
		values[key] = value
	}
	return values
}

func TestLoadModelsWithTheDefaultOnly(t *testing.T) {
	getenv, environ := modelEnv(nil)
	models, err := loadModels(getenv, environ)
	if err != nil {
		t.Fatal(err)
	}
	// An installation of one model loads it as before, under the ID
	// default and its name for label.
	want, err := loadLLM(getenv)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "default" || models[0].Label != "gpt-luna" || !reflect.DeepEqual(models[0].LLM, want) {
		t.Fatalf("models = %+v", models)
	}

	getenv, environ = modelEnv(map[string]string{LLMLabelEnv: "GPT Luna"})
	if models, err = loadModels(getenv, environ); err != nil || models[0].Label != "GPT Luna" {
		t.Fatalf("labelled default = %+v, %v", models, err)
	}
}

// A model name longer than a label, such as the ARN of an inference
// profile, is cut for its label: the configuration starts as before.
func TestLoadModelsCutsALongModelNameForItsLabel(t *testing.T) {
	name := "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/" + strings.Repeat("é", 10)
	getenv, environ := modelEnv(map[string]string{"OPENSVC_AI_LLM_MODEL": name})
	models, err := loadModels(getenv, environ)
	if err != nil || models[0].LLM.Model != name || models[0].Label != string([]rune(name)[:64]) {
		t.Fatalf("models = %+v, %v", models, err)
	}
	// A label given is not cut.
	getenv, environ = modelEnv(map[string]string{"OPENSVC_AI_LLM_MODEL": name, LLMLabelEnv: name})
	if _, err := loadModels(getenv, environ); err == nil || !strings.Contains(err.Error(), LLMLabelEnv) {
		t.Fatalf("a long label error = %v", err)
	}
}

func TestLoadModelsAddsTheModelsInTheirOrder(t *testing.T) {
	getenv, environ := modelEnv(withEnv(haikuEnv, map[string]string{
		ModelsEnv:                              " haiku , mini ",
		"OPENSVC_AI_MODEL_MINI_PROTOCOL":       "chat_completions",
		"OPENSVC_AI_MODEL_MINI_BASE_URL":       "https://api.openai.example.test/v1",
		"OPENSVC_AI_MODEL_MINI_MODEL":          "gpt-luna-mini",
		"OPENSVC_AI_MODEL_MINI_AUTH_MODE":      "bearer",
		"OPENSVC_AI_MODEL_MINI_API_TOKEN_FROM": "default",
		"OPENSVC_AI_MODEL_MINI_STORE":          "omit",
		"OPENSVC_AI_MODEL_MINI_TIMEOUT":        "30s",
	}))
	models, err := loadModels(getenv, environ)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 3 || models[0].ID != "default" || models[1].ID != "haiku" || models[2].ID != "mini" {
		t.Fatalf("models = %+v", models)
	}
	haiku, mini := models[1], models[2]
	if haiku.Label != "Claude Haiku" || haiku.LLM.Protocol != LLMProtocolMessages || haiku.LLM.Model != "claude-haiku" ||
		haiku.LLM.AuthMode != LLMAuthModeAPIKey || haiku.LLM.AuthHeader != "x-api-key" || haiku.LLM.APITokenEnv != "OPENSVC_AI_MODEL_HAIKU_API_TOKEN" {
		t.Fatalf("haiku = %+v", haiku)
	}
	// A model shares the key of another by the name of its variable, and
	// reads none of the settings of the default.
	if mini.Label != "gpt-luna-mini" || mini.LLM.APITokenEnv != LLMAPITokenEnv || !mini.LLM.OmitStore || mini.LLM.Timeout.String() != "30s" ||
		mini.LLM.MaxOutputTokens != DefaultMaxOutputTokens {
		t.Fatalf("mini = %+v", mini)
	}
	for _, model := range models {
		if strings.Contains(model.Label+model.LLM.BaseURL+model.LLM.Model+model.LLM.APITokenEnv+model.LLM.AuthHeader, "-key-") ||
			strings.Contains(model.LLM.APITokenEnv, "haiku-key") || strings.Contains(model.LLM.APITokenEnv, "default-key") {
			t.Fatalf("model %s retained a key: %+v", model.ID, model)
		}
	}
}

func TestLoadModelsRejectsInvalidModels(t *testing.T) {
	for _, test := range []struct {
		name string
		env  map[string]string
		want string
	}{
		{name: "the default", env: map[string]string{ModelsEnv: "default"}, want: "names the model of the OPENSVC_AI_LLM_* variables"},
		{name: "a capital", env: map[string]string{ModelsEnv: "Haiku"}, want: "must be a lowercase letter"},
		{name: "a dash", env: map[string]string{ModelsEnv: "gpt-luna"}, want: "must be a lowercase letter"},
		{name: "an underscore", env: map[string]string{ModelsEnv: "gpt_luna"}, want: "must be a lowercase letter"},
		{name: "a digit first", env: map[string]string{ModelsEnv: "4o"}, want: "must be a lowercase letter"},
		{name: "too long", env: map[string]string{ModelsEnv: "abcdefghijklmnopq"}, want: "must be a lowercase letter"},
		{name: "an empty entry", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku,"}), want: "must be a lowercase letter"},
		{name: "repeated", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku,haiku"}), want: "is repeated"},
		{name: "too many", env: map[string]string{ModelsEnv: "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p"}, want: "at most 15 models"},
		{name: "a missing variable", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_BASE_URL": ""}), want: "OPENSVC_AI_MODEL_HAIKU_BASE_URL is required"},
		{name: "a missing key", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN": ""}), want: "OPENSVC_AI_MODEL_HAIKU_API_TOKEN is required"},
		{name: "an invalid value", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_REASONING_EFFORT": "high"}), want: "OPENSVC_AI_MODEL_HAIKU_REASONING_EFFORT requires"},
		{name: "an invalid label", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_LABEL": strings.Repeat("a", 65)}), want: "OPENSVC_AI_MODEL_HAIKU_LABEL"},
		{name: "an invalid default label", env: map[string]string{LLMLabelEnv: "a\tb"}, want: LLMLabelEnv},
		{name: "a model not listed", env: haikuEnv, want: `OPENSVC_AI_MODEL_HAIKU_API_TOKEN: model "haiku" is not in OPENSVC_AI_MODELS: list it there, or remove its variables`},
		{name: "a model in another case", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_Haiku_TIMEOUT": "1m"}), want: "is not in OPENSVC_AI_MODELS"},
		{name: "an unknown variable", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_BASEURL": "x"}), want: `OPENSVC_AI_MODEL_HAIKU_BASEURL: unknown variable of model "haiku"`},
		{name: "a key and a shared key", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN_FROM": "default"}), want: "are exclusive"},
		{name: "a shared key of no model", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN": "", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN_FROM": "luna"}), want: `API_TOKEN_FROM "luna" is not a model`},
		{name: "a shared key of itself", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN": "", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN_FROM": "haiku"}), want: "with its own API_TOKEN"},
		{name: "a shared key without authentication", env: withEnv(haikuEnv, map[string]string{ModelsEnv: "haiku", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN": "", "OPENSVC_AI_MODEL_HAIKU_AUTH_MODE": "none", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN_FROM": "default"}), want: "requires an authentication"},
		{name: "a shared shared key", env: withEnv(haikuEnv, map[string]string{
			ModelsEnv: "haiku,mini", "OPENSVC_AI_MODEL_MINI_PROTOCOL": "responses", "OPENSVC_AI_MODEL_MINI_BASE_URL": "https://x.example.test/v1",
			"OPENSVC_AI_MODEL_MINI_MODEL": "mini", "OPENSVC_AI_MODEL_MINI_AUTH_MODE": "bearer", "OPENSVC_AI_MODEL_MINI_API_TOKEN_FROM": "haiku",
			"OPENSVC_AI_MODEL_HAIKU_API_TOKEN": "", "OPENSVC_AI_MODEL_HAIKU_API_TOKEN_FROM": "default",
		}), want: `API_TOKEN_FROM "haiku" must name a model with its own API_TOKEN`},
		{name: "an invalid default", env: map[string]string{"OPENSVC_AI_LLM_MODEL": ""}, want: "OPENSVC_AI_LLM_MODEL is required"},
		{name: "a shared key of the default", env: map[string]string{"OPENSVC_AI_LLM_API_TOKEN_FROM": "haiku"}, want: "OPENSVC_AI_LLM_API_TOKEN_FROM is not read"},
	} {
		t.Run(test.name, func(t *testing.T) {
			getenv, environ := modelEnv(test.env)
			_, err := loadModels(getenv, environ)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("loadModels() error = %v, want containing %q", err, test.want)
			}
		})
	}
}
