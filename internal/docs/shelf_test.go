package docs

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestLineOfAVersion(t *testing.T) {
	for version, want := range map[string]string{
		"v3.0.0-rc56-41-g451250d44": "3.0",
		"v3.1.0-rc2":                "3.1",
		"3.12.4":                    "3.12",
		"3.0":                       "3.0",
		"OpenSVC agent v3.0.0-rc56": "3.0",
		"":                          "",
		"unknown":                   "",
	} {
		if got := Line(version); got != want {
			t.Errorf("Line(%q) = %q, want %q", version, got, want)
		}
	}
	if !OlderLine("3.0", "3.1") || !OlderLine("2.9", "3.0") || !OlderLine("3.2", "3.10") || OlderLine("3.1", "3.1") || OlderLine("3.1", "3.0") {
		t.Error("OlderLine misorders release lines")
	}
}

// site serves a documentation under /lab/: its versions.json, when set, and
// the llms.txt of its latest book, whose build line it gives.
type site struct {
	mu       sync.Mutex
	versions string
	build    string
	down     bool
}

func (s *site) set(versions, build string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.versions, s.build = versions, build
}

func (s *site) serve(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		defer s.mu.Unlock()
		if s.down {
			http.Error(w, "down", http.StatusServiceUnavailable)
			return
		}
		switch r.URL.Path {
		case "/lab/versions.json":
			if s.versions == "" {
				http.NotFound(w, r)
				return
			}
			_, _ = w.Write([]byte(s.versions))
		case "/lab/llms.txt":
			_, _ = w.Write([]byte("# OpenSVC\n\nBuild: " + s.build + ".\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

// books is a loader of the books of the indexes it holds, the release of
// each, counting the reads of each index.
type books struct {
	mu      sync.Mutex
	library map[string]*Library
	release map[string]string
	reads   map[string]int
}

func newBooks() *books {
	return &books{library: map[string]*Library{}, release: map[string]string{}, reads: map[string]int{}}
}

func (b *books) add(t *testing.T, index, release, text string) *Library {
	t.Helper()
	library, err := New([]Page{{Path: "agent/install.md", Text: "# Install\n\n" + text + "\n"}}, index)
	if err != nil {
		t.Fatal(err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.library[index], b.release[index] = library, release
	return library
}

func (b *books) load(_ context.Context, index string) (*Library, string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.reads[index]++
	if library, ok := b.library[index]; ok {
		return library, b.release[index], nil
	}
	return nil, "", errors.New(index + ": 404 Not Found")
}

func (b *books) count(index string) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.reads[index]
}

// open makes the shelf of root and reads it, as the agent does at start;
// logs keeps what it says.
func open(t *testing.T, client *http.Client, root string, load Loader) (*Shelf, *[]string, error) {
	t.Helper()
	var (
		mu   sync.Mutex
		logs []string
	)
	s, err := NewShelf(root, client, load, func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		logs = append(logs, fmt.Sprintf(format, args...))
	})
	if err != nil {
		t.Fatal(err)
	}
	return s, &logs, s.Refresh(context.Background())
}

const twoLines = `[{"name": "v3.1", "path": "/"}, {"name": "v3.0", "path": "/v3.0/"}]`

func TestShelfGivesTheBookOfTheReleaseLineAsked(t *testing.T) {
	documentation := &site{versions: twoLines, build: "abc 2026-11-03"}
	server := documentation.serve(t)
	b := newBooks()
	latest := b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.1.0", "The latest book.")
	earlier := b.add(t, server.URL+"/lab/v3.0/llms.txt", "OpenSVC agent v3.0.0-rc56", "The v3.0 book.")
	// The root may name its llms.txt.
	s, _, err := open(t, server.Client(), server.URL+"/lab/llms.txt", b.load)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Lines(); !slices.Equal(got, []string{"3.1", "3.0"}) {
		t.Errorf("lines = %v", got)
	}
	for _, c := range []struct {
		version string
		book    *Library
		says    string
	}{
		{"3.1", latest, "the book of the 3.1 release line, the one of 3.1"},
		{"v3.0.0-rc56-41-g451250d44", earlier, "the book of the 3.0 release line, the one of v3.0.0-rc56-41-g451250d44"},
		{"3.0", earlier, "the book of the 3.0 release line"},
		{"v3.2.0", latest, "newer than the latest book, 3.1"},
		{"2.1", latest, "No book of the 2.1 release line, the one of 2.1, is published"},
		{"", latest, "No release line was asked: this is the book of the latest one, 3.1"},
		{"latest", latest, `"latest" names no release line`},
	} {
		book, says := s.Book(context.Background(), c.version)
		if book != c.book || !strings.Contains(says, c.says) {
			t.Errorf("Book(%q) = %s, %q; want %s, %q", c.version, book.Source(), says, c.book.Source(), c.says)
		}
	}
	// The latest book is read at start, the v3.0 one once, on demand.
	if n, m := b.count(server.URL+"/lab/llms.txt"), b.count(server.URL+"/lab/v3.0/llms.txt"); n != 1 || m != 1 {
		t.Errorf("reads of the latest book %d, of the v3.0 one %d, want 1 and 1", n, m)
	}
}

func TestShelfWithNoVersionsIsOfTheLineOfItsKeywordReference(t *testing.T) {
	documentation := &site{build: "abc"}
	server := documentation.serve(t)
	b := newBooks()
	latest := b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.0.0-rc56", "The only book.")
	s, _, err := open(t, server.Client(), server.URL+"/lab", b.load)
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Lines(); !slices.Equal(got, []string{"3.0"}) {
		t.Errorf("lines = %v", got)
	}
	if book, says := s.Book(context.Background(), "v3.0.0-rc53"); book != latest || !strings.Contains(says, "the book of the 3.0 release line") {
		t.Errorf("Book = %s, %q", book.Source(), says)
	}
}

func TestShelfLeavesOutTheVersionsItCannotServe(t *testing.T) {
	for _, c := range []struct {
		versions string
		lines    []string
		says     string
	}{
		{`[{"name": "v3.1", "path": "/"}, {"name": "v3.0", "path": "/../"}, {"name": "latest", "path": "/v9/"}]`, []string{"3.1"}, `"v3.0" at "/../" is no release line under the root`},
		{`[{"name": "v3.1", "path": "/"}, {"name": "v3.2", "path": "/"}, {"name": "v3.0", "path": "/v3.0/"}]`, []string{"3.1", "3.0"}, `"v3.2" is a second version at /`},
		{`[{"name": "v2.1", "path": "/v2.1/"}]`, []string{"3.1", "2.1"}, "lists no version at /: the root is the book of its keyword reference"},
		{`{"v3.1": "/"}`, []string{"3.1"}, "is no list of versions"},
	} {
		documentation := &site{versions: c.versions, build: "abc"}
		server := documentation.serve(t)
		b := newBooks()
		latest := b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.1.0", "The latest book.")
		s, logs, err := open(t, server.Client(), server.URL+"/lab/", b.load)
		if err != nil {
			t.Fatalf("%s: %v", c.versions, err)
		}
		if !slices.Equal(s.Lines(), c.lines) || s.Latest() != latest || !strings.Contains(strings.Join(*logs, "\n"), c.says) {
			t.Errorf("%s: lines %v, logs %q", c.versions, s.Lines(), *logs)
		}
	}
	if _, err := NewShelf("http://book.example/", http.DefaultClient, newBooks().load, nil); err == nil || !strings.Contains(err.Error(), "is not an https URL") {
		t.Errorf("http root: err = %v", err)
	}
}

// clock is a time a test moves.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) get() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) add(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// settled waits for the refresh a search started in the background.
func settled(t *testing.T, s *Shelf) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); time.Sleep(5 * time.Millisecond) {
		s.mu.Lock()
		done := !s.refreshing
		s.mu.Unlock()
		if done {
			return
		}
	}
	t.Fatal("the refresh never ended")
}

func TestShelfReadsItsBooksAgainWhenTheyChange(t *testing.T) {
	documentation := &site{versions: twoLines, build: "abc"}
	server := documentation.serve(t)
	b := newBooks()
	b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.1.0", "The first build.")
	b.add(t, server.URL+"/lab/v3.0/llms.txt", "OpenSVC agent v3.0.0-rc56", "The v3.0 book.")
	s, _, err := open(t, server.Client(), server.URL+"/lab/", b.load)
	if err != nil {
		t.Fatal(err)
	}
	c := &clock{now: time.Now()}
	s.now = c.get
	s.Book(context.Background(), "3.0")

	// Within shelfRefresh, nothing is read again; past it, an unchanged
	// documentation is checked, its books not read again.
	s.Book(context.Background(), "3.1")
	c.add(shelfRefresh)
	s.Book(context.Background(), "3.1")
	settled(t, s)
	if n := b.count(server.URL + "/lab/llms.txt"); n != 1 {
		t.Errorf("reads of an unchanged latest book = %d, want 1", n)
	}

	// A version listed, the build of the latest book unchanged, changes the
	// lines, not the latest book.
	documentation.set(`[{"name": "v3.1", "path": "/"}, {"name": "v3.0", "path": "/v3.0/"}, {"name": "v2.1", "path": "/v2.1/"}]`, "abc")
	c.add(shelfRefresh)
	s.Book(context.Background(), "3.1")
	settled(t, s)
	if n := b.count(server.URL + "/lab/llms.txt"); n != 1 || !slices.Equal(s.Lines(), []string{"3.1", "3.0", "2.1"}) {
		t.Errorf("after a version listed: reads of the latest book %d, want 1; lines %v", n, s.Lines())
	}

	// A new release line moves the latest book, read again, and the 3.1 one
	// under its own path, read when asked.
	documentation.set(`[{"name": "v3.2", "path": "/"}, {"name": "v3.1", "path": "/v3.1/"}, {"name": "v3.0", "path": "/v3.0/"}]`, "def")
	second := b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.2.0", "The second build.")
	moved := b.add(t, server.URL+"/lab/v3.1/llms.txt", "OpenSVC agent v3.1.4", "The v3.1 book.")
	c.add(shelfRefresh)
	s.Book(context.Background(), "3.1")
	settled(t, s)
	if s.Latest() != second || !slices.Equal(s.Lines(), []string{"3.2", "3.1", "3.0"}) {
		t.Fatalf("after the release, latest %s, lines %v", s.Latest().Source(), s.Lines())
	}
	if book, says := s.Book(context.Background(), "v3.1.4"); book != moved || !strings.Contains(says, "the 3.1 release line") {
		t.Errorf("Book(v3.1.4) = %s, %q", book.Source(), says)
	}
	if book, _ := s.Book(context.Background(), "3.2"); book != second {
		t.Errorf("Book(3.2) = %s", book.Source())
	}
	// A new build of the latest book, the versions unchanged, is read too.
	documentation.set(`[{"name": "v3.2", "path": "/"}, {"name": "v3.1", "path": "/v3.1/"}, {"name": "v3.0", "path": "/v3.0/"}]`, "ghi")
	third := b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.2.1", "The third build.")
	c.add(shelfRefresh)
	s.Book(context.Background(), "3.2")
	settled(t, s)
	if s.Latest() != third {
		t.Errorf("after a new build, latest %s", s.Latest().Source())
	}
}

func TestShelfAsksAgainABookItCouldNotRead(t *testing.T) {
	documentation := &site{versions: twoLines, build: "abc"}
	server := documentation.serve(t)
	b := newBooks()
	b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.1.0", "The latest book.")
	s, _, err := open(t, server.Client(), server.URL+"/lab/", b.load)
	if err != nil {
		t.Fatal(err)
	}
	c := &clock{now: time.Now()}
	s.now = c.get
	index := server.URL + "/lab/v3.0/llms.txt"
	for range 2 {
		if _, says := s.Book(context.Background(), "v3.0.0"); !strings.Contains(says, "could not be read") {
			t.Fatalf("says %q", says)
		}
	}
	if n := b.count(index); n != 1 {
		t.Fatalf("reads = %d within the retry delay, want 1", n)
	}
	c.add(shelfRetry)
	s.Book(context.Background(), "v3.0.0")
	if n := b.count(index); n != 2 {
		t.Errorf("reads = %d past the retry delay, want 2", n)
	}
}

func TestShelfUnreadAtStartIsReadOnTheNextSearches(t *testing.T) {
	documentation := &site{versions: twoLines, build: "abc", down: true}
	server := documentation.serve(t)
	b := newBooks()
	latest := b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.1.0", "The latest book.")
	s, _, err := open(t, server.Client(), server.URL+"/lab/", b.load)
	if err == nil || s.Latest() != nil {
		t.Fatalf("a documentation down: err %v, latest %v", err, s.Latest())
	}
	c := &clock{now: time.Now()}
	s.now = c.get
	if book, says := s.Book(context.Background(), "3.1"); book != nil || !strings.Contains(says, "could not be read") {
		t.Errorf("Book = %v, %q", book, says)
	}
	// Up again, it is read within shelfUnreadRetry, on a search.
	documentation.mu.Lock()
	documentation.down = false
	documentation.mu.Unlock()
	c.add(shelfUnreadRetry)
	s.Book(context.Background(), "3.1")
	settled(t, s)
	if book, _ := s.Book(context.Background(), "3.1"); book != latest || !slices.Equal(s.Lines(), []string{"3.1", "3.0"}) {
		t.Errorf("after the documentation is up: book %v, lines %v", book, s.Lines())
	}
}

// TestShelfSurvivesABookItCannotIndex runs a loader that panics, as an
// indexing that did not foresee a page would: on the refresh in the
// background, the latest book stays, said in the log; on the book of an
// earlier line, the search takes the latest book. The process goes on.
func TestShelfSurvivesABookItCannotIndex(t *testing.T) {
	documentation := &site{versions: twoLines, build: "abc"}
	server := documentation.serve(t)
	b := newBooks()
	first := b.add(t, server.URL+"/lab/llms.txt", "OpenSVC agent v3.1.0", "The first build.")
	var mu sync.Mutex
	panics := false
	load := func(ctx context.Context, index string) (*Library, string, error) {
		mu.Lock()
		p := panics
		mu.Unlock()
		if p {
			panic("index out of range [-1]")
		}
		return b.load(ctx, index)
	}
	s, logs, err := open(t, server.Client(), server.URL+"/lab/", load)
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	panics = true
	mu.Unlock()
	if book, says := s.Book(context.Background(), "3.0"); book != first || !strings.Contains(says, "could not be read") {
		t.Errorf("Book(3.0) = %s, %q", book.Source(), says)
	}
	documentation.set(twoLines, "def")
	c := &clock{now: time.Now()}
	s.mu.Lock()
	s.now = c.get
	s.mu.Unlock()
	c.add(shelfRefresh)
	s.Book(context.Background(), "3.1")
	settled(t, s)
	if s.Latest() != first || !strings.Contains(strings.Join(*logs, "\n"), "panic: index out of range [-1]") {
		t.Errorf("after a refresh that panicked: latest %s, logs %q", s.Latest().Source(), *logs)
	}
}

func TestOlderVersionComparesTheWholeVersion(t *testing.T) {
	for _, c := range []struct {
		a, b  string
		older bool
	}{
		{"v3.0.0-rc46-12-g1234567", "v3.0.0-rc47", true},
		{"v3.0.0-rc56-41-g451250d44", "v3.0.0-rc56-3-gabcdef0", false},
		{"v3.0.0-rc56-3-gabcdef0", "v3.0.0-rc56-41-g451250d44", true},
		{"v3.0.0-rc56", "v3.0.0-rc56-1-gabcdef0", true},
		{"v3.0.0-rc9", "v3.0.0-rc10", true},
		{"v3.0.0-rc57", "v3.0.0", true},
		{"v3.0.0", "v3.0.0-rc57", false},
		{"v3.0.4", "v3.0.10", true},
		{"v2.1.63", "v3.0.0-rc1", true},
		{"v3.1.0", "v3.0.9", false},
		{"v3.0.0-rc56", "v3.0.0-rc56", false},
		{"2.1.63", "3.0.4", true},
	} {
		if got := OlderVersion(c.a, c.b); got != c.older {
			t.Errorf("OlderVersion(%q, %q) = %v, want %v", c.a, c.b, got, c.older)
		}
	}
}
