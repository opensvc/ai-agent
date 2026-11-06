package config

import (
	"testing"
	"time"
)

func processEnv(values map[string]string) func(string) string {
	return func(key string) string {
		if value, ok := values[key]; ok {
			return value
		}
		switch key {
		case "OPENSVC_AI_TLS_CERT_FILE":
			return "/etc/opensvc-ai/agent.crt"
		case "OPENSVC_AI_TLS_KEY_FILE":
			return "/etc/opensvc-ai/agent.key"
		}
		return ""
	}
}

func TestLoadHTTPS(t *testing.T) {
	for _, tc := range []struct {
		name, address string
		wantErr       bool
	}{
		{"default", "", false}, {"ipv4", "0.0.0.0:8090", false}, {"ipv6", "[::1]:8090", false},
		{"missing IP", ":8090", true}, {"hostname", "localhost:8090", true},
		{"missing port", "127.0.0.1", true}, {"zero", "127.0.0.1:0", true},
		{"large port", "127.0.0.1:65536", true}, {"named port", "127.0.0.1:https", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := load(processEnv(map[string]string{"OPENSVC_AI_LISTEN_ADDR": tc.address}))
			if tc.wantErr {
				if err == nil {
					t.Fatal("invalid address succeeded")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := tc.address
			if want == "" {
				want = DefaultListenAddress
			}
			if cfg.ListenAddress != want || cfg.TLSCertFile != "/etc/opensvc-ai/agent.crt" || cfg.TLSKeyFile != "/etc/opensvc-ai/agent.key" || cfg.MaxConcurrentAsks != DefaultMaxConcurrentAsks || cfg.ShutdownTimeout != DefaultShutdownTimeout {
				t.Fatalf("unexpected config: %+v", cfg)
			}
		})
	}
	for _, key := range []string{"OPENSVC_AI_TLS_CERT_FILE", "OPENSVC_AI_TLS_KEY_FILE"} {
		for _, value := range []string{"", "/", "relative.pem"} {
			if _, err := load(processEnv(map[string]string{key: value})); err == nil {
				t.Fatalf("%s=%q succeeded", key, value)
			}
		}
	}
}

func TestLoadShutdownTimeout(t *testing.T) {
	config, err := load(func(key string) string {
		if key == "OPENSVC_AI_SHUTDOWN_TIMEOUT" {
			return "45s"
		}
		return processEnv(nil)(key)
	})
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if config.ShutdownTimeout != 45*time.Second {
		t.Fatalf("got shutdown timeout %s, want 45s", config.ShutdownTimeout)
	}

	for _, value := range []string{"invalid", "0s", "500ms", "6m"} {
		_, err := load(func(key string) string {
			if key == "OPENSVC_AI_SHUTDOWN_TIMEOUT" {
				return value
			}
			return processEnv(nil)(key)
		})
		if err == nil {
			t.Fatalf("value %q succeeded", value)
		}
	}
}

func TestLoadMaxConcurrentAsksPerUser(t *testing.T) {
	env := func(asks, perUser string) func(string) string {
		return func(key string) string {
			switch key {
			case "OPENSVC_AI_MAX_CONCURRENT_ASKS":
				return asks
			case "OPENSVC_AI_MAX_CONCURRENT_ASKS_PER_USER":
				return perUser
			}
			return processEnv(nil)(key)
		}
	}
	for _, test := range []struct {
		asks, perUser string
		want          int
	}{{"", "", 2}, {"1", "", 1}, {"8", "3", 3}, {"8", "8", 8}} {
		config, err := load(env(test.asks, test.perUser))
		if err != nil || config.MaxConcurrentAsksPerUser != test.want {
			t.Errorf("asks %q, per user %q: %d, %v, want %d", test.asks, test.perUser, config.MaxConcurrentAsksPerUser, err, test.want)
		}
	}
	for _, perUser := range []string{"0", "5", "x"} {
		if _, err := load(env("4", perUser)); err == nil {
			t.Errorf("per user %q accepted", perUser)
		}
	}
}

func TestLoadMaxConcurrentAsks(t *testing.T) {
	config, err := load(func(key string) string {
		if key == "OPENSVC_AI_MAX_CONCURRENT_ASKS" {
			return "12"
		}
		return processEnv(nil)(key)
	})
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if config.MaxConcurrentAsks != 12 {
		t.Fatalf("got max concurrent asks %d, want 12", config.MaxConcurrentAsks)
	}

	for _, value := range []string{"invalid", "0", "129"} {
		_, err := load(func(key string) string {
			if key == "OPENSVC_AI_MAX_CONCURRENT_ASKS" {
				return value
			}
			return processEnv(nil)(key)
		})
		if err == nil {
			t.Fatalf("value %q succeeded", value)
		}
	}
}
