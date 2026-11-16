package llm

import (
	"net/http"
	"testing"
	"time"
)

func TestRetryAfter(t *testing.T) {
	now := time.Date(2026, 11, 16, 12, 0, 0, 0, time.UTC)
	for value, want := range map[string]time.Duration{
		"":                              0,
		"3":                             3 * time.Second,
		"0":                             0,
		"-1":                            0,
		"1.5":                           0,
		"NaN":                           0,
		"999999999999":                  time.Hour,
		"7200":                          time.Hour,
		"99999999999999999999":          time.Hour,
		"-99999999999999999999":         0,
		"Tue, 17 Nov 2026 12:00:00 GMT": time.Hour,
		"Mon, 16 Nov 2026 12:00:30 GMT": 30 * time.Second,
		"Mon, 16 Nov 2026 11:59:00 GMT": 0,
		"soon":                          0,
	} {
		header := http.Header{}
		if value != "" {
			header.Set("Retry-After", value)
		}
		if got := RetryAfter(header, now); got != want {
			t.Errorf("RetryAfter(%q) = %v, want %v", value, got, want)
		}
	}
}

func TestTransientStatus(t *testing.T) {
	for _, status := range []int{408, 429, 500, 502, 503, 504, 529} {
		if !TransientStatus(status) {
			t.Errorf("status %d not transient", status)
		}
	}
	for _, status := range []int{200, 400, 401, 403, 404, 413, 422} {
		if TransientStatus(status) {
			t.Errorf("status %d transient", status)
		}
	}
}
