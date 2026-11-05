package docs

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeCode serves the code of the refs it holds, counting the reads; a ref
// of hold waits on its channel.
type fakeCode struct {
	mu    sync.Mutex
	refs  map[string]bool
	reads map[string]int
	panic bool
	hold  map[string]chan struct{}
}

func (f *fakeCode) fetch(_ context.Context, ref string) ([]File, error) {
	f.mu.Lock()
	wait := f.hold[ref]
	f.mu.Unlock()
	if wait != nil {
		<-wait
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.reads[ref]++
	if f.panic {
		panic("index out of range [-1]")
	}
	if !f.refs[ref] {
		return nil, fmt.Errorf("read %s: %w", ref, errNotPublished)
	}
	return []File{{Path: "daemon/nmon/speaker.go", Text: speakerFile}}, nil
}

func (f *fakeCode) count(ref string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.reads[ref]
}

// codeShelf makes a shelf of the code f serves, its log kept in logs.
func codeShelf(f *fakeCode) (*CodeShelf, *[]string) {
	var (
		mu   sync.Mutex
		logs []string
	)
	s := NewCodeShelf(nil, "https://code.example/{ref}", "main", func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		logs = append(logs, fmt.Sprintf(format, args...))
	})
	s.fetch = f.fetch
	return s, &logs
}

func TestCodeShelfGivesTheCodeOfTheCommitElseOfItsTag(t *testing.T) {
	f := &fakeCode{refs: map[string]bool{"451250d44": true, "v3.0.0-rc56": true}, reads: map[string]int{}}
	s, logs := codeShelf(f)
	// A version read is kept: asked again, it is not read again.
	s.Code(context.Background(), "v3.0.0-rc56-41-g451250d44")
	s.Code(context.Background(), "v3.0.0-rc56-41-g451250d44")
	if n := f.count("451250d44"); n != 1 {
		t.Errorf("reads of 451250d44 = %d, want 1", n)
	}
	for _, c := range []struct {
		version, says string
		found         bool
	}{
		{"v3.0.0-rc56-41-g451250d44", "This is the code of v3.0.0-rc56-41-g451250d44.", true},
		{"v3.0.0-rc56-2-gabcdef0", "is not published: this is the code of its tag v3.0.0-rc56, which is not exactly the code that runs", true},
		{"v3.0.0-rc57-1-g1234567", "No code is published at v3.0.0-rc57-1-g1234567: name a tag in full", false},
		// The tags carry a v, given a version without it.
		{"3.0.0-rc56", "This is the code of v3.0.0-rc56.", true},
		{"latest", `"latest" names no version of the code: name a tag in full`, false},
	} {
		index, says := s.Code(context.Background(), c.version)
		if (index != nil) != c.found || !strings.Contains(says, c.says) {
			t.Errorf("Code(%q) = %v, %q; want %q", c.version, index != nil, says, c.says)
		}
	}
	// Each read is told the log: what was read, the tag read in place of a
	// commit, and why a read failed.
	log := strings.Join(*logs, "\n")
	for _, want := range []string{
		"code: v3.0.0-rc56-41-g451250d44 read at 451250d44, 1 files, 2 declarations, 0 files left out, in",
		"code: v3.0.0-rc56-2-gabcdef0 is not published at abcdef0: its tag v3.0.0-rc56 is read",
		"code: read v3.0.0-rc57-1-g1234567 at v3.0.0-rc57: read v3.0.0-rc57: not published",
	} {
		if !strings.Contains(log, want) {
			t.Errorf("log lacks %q:\n%s", want, log)
		}
	}
}

func TestCodeShelfKeepsThreeVersionsAndRetriesAFailedRead(t *testing.T) {
	f := &fakeCode{refs: map[string]bool{"v3.0.1": true, "v3.0.2": true, "v3.0.3": true, "v3.0.4": true}, reads: map[string]int{}}
	s, _ := codeShelf(f)
	clock := time.Now()
	s.now = func() time.Time { return clock }
	s.Code(context.Background(), "v3.0.1")
	s.Code(context.Background(), "v3.0.2")
	s.Code(context.Background(), "v3.0.1")
	s.Code(context.Background(), "v3.0.3")
	s.Code(context.Background(), "v3.0.4")
	// v3.0.2, the least recently used, left: it is read again, v3.0.1 not.
	s.Code(context.Background(), "v3.0.1")
	s.Code(context.Background(), "v3.0.2")
	if a, b := f.count("v3.0.1"), f.count("v3.0.2"); a != 1 || b != 2 {
		t.Errorf("reads of v3.0.1 %d, of v3.0.2 %d; want 1 and 2", a, b)
	}
	s.mu.Lock()
	n := len(s.entries)
	s.mu.Unlock()
	if n != maxCodeIndexes {
		t.Errorf("entries = %d, want %d", n, maxCodeIndexes)
	}
	// A failed read is asked again after shelfRetry only.
	s.Code(context.Background(), "v9.9.9")
	s.Code(context.Background(), "v9.9.9")
	if n := f.count("v9.9.9"); n != 1 {
		t.Errorf("reads of a failed version = %d within the retry delay, want 1", n)
	}
	clock = clock.Add(shelfRetry)
	s.Code(context.Background(), "v9.9.9")
	if n := f.count("v9.9.9"); n != 2 {
		t.Errorf("reads of a failed version = %d past the retry delay, want 2", n)
	}
}

func TestCodeShelfSurvivesAReadThatPanics(t *testing.T) {
	s, logs := codeShelf(&fakeCode{refs: map[string]bool{"v3.0.1": true}, reads: map[string]int{}, panic: true})
	defer func() {
		if !strings.Contains(strings.Join(*logs, "\n"), "panic: index out of range [-1]") {
			t.Errorf("logs %q", *logs)
		}
	}()
	if index, says := s.Code(context.Background(), "v3.0.1"); index != nil || !strings.Contains(says, "could not be read") {
		t.Errorf("Code = %v, %q", index, says)
	}
}

// TestCodeShelfKeepsAReadInProgress evicts no read still in progress, the
// searches waiting on it: the shelf holds more than maxCodeIndexes meanwhile.
func TestCodeShelfKeepsAReadInProgress(t *testing.T) {
	release := make(chan struct{})
	f := &fakeCode{refs: map[string]bool{"v3.0.1": true, "v3.0.2": true, "v3.0.3": true}, reads: map[string]int{}, hold: map[string]chan struct{}{"v3.0.1": release}}
	s, _ := codeShelf(f)
	done := make(chan *CodeIndex)
	go func() {
		index, _ := s.Code(context.Background(), "v3.0.1")
		done <- index
	}()
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(5 * time.Millisecond) {
		s.mu.Lock()
		_, started := s.entries["v3.0.1"]
		s.mu.Unlock()
		if started {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the read of v3.0.1 never started")
		}
	}
	s.Code(context.Background(), "v3.0.2")
	s.Code(context.Background(), "v3.0.3")
	s.mu.Lock()
	_, kept := s.entries["v3.0.1"]
	s.mu.Unlock()
	close(release)
	if index := <-done; !kept || index == nil {
		t.Errorf("the read in progress kept %v, served %v", kept, index != nil)
	}
	if n := f.count("v3.0.1"); n != 1 {
		t.Errorf("reads of v3.0.1 = %d, want 1", n)
	}
}

// TestCodeShelfReadsTheLatestCodeWithNoVersion reads, for a search naming no
// version, the latest code, at its branch, again past its lifetime.
func TestCodeShelfReadsTheLatestCodeWithNoVersion(t *testing.T) {
	f := &fakeCode{refs: map[string]bool{"main": true}, reads: map[string]int{}}
	s, logs := codeShelf(f)
	var clockMu sync.Mutex
	clock := time.Date(2026, 11, 4, 12, 0, 0, 0, time.UTC)
	s.now = func() time.Time { clockMu.Lock(); defer clockMu.Unlock(); return clock }
	index, says := s.Code(context.Background(), "")
	if index == nil || says != "This is the latest code of om3, at main, read 2026-11-04 12:00 UTC: a cluster may run another version." || s.Latest() != "main" {
		t.Fatalf("Code() = %v, %q", index != nil, says)
	}
	s.Code(context.Background(), "")
	if n := f.count("main"); n != 1 {
		t.Errorf("reads of main within its lifetime = %d, want 1", n)
	}
	// Past its lifetime, the code is read again, the searches served the
	// code read before until the new read ends.
	hold := make(chan struct{})
	f.mu.Lock()
	f.hold = map[string]chan struct{}{"main": hold}
	f.mu.Unlock()
	clockMu.Lock()
	clock = clock.Add(codeLatestLifetime)
	clockMu.Unlock()
	if index, says := s.Code(context.Background(), ""); index == nil || !strings.Contains(says, "read 2026-11-04 12:00 UTC") {
		t.Errorf("Code() while the latest code is read again = %v, %q", index != nil, says)
	}
	s.Code(context.Background(), "")
	close(hold)
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, says := s.Code(context.Background(), ""); strings.Contains(says, "read 2026-11-04 18:00 UTC") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the latest code read again never served")
		}
		time.Sleep(time.Millisecond)
	}
	if n := f.count("main"); n != 2 {
		t.Errorf("reads of main past its lifetime = %d, want 2", n)
	}
	if !strings.Contains(strings.Join(*logs, "\n"), "code: the latest code read at main") {
		t.Errorf("logs %q", *logs)
	}
	// A latest code that cannot be read says so.
	s2, _ := codeShelf(&fakeCode{refs: map[string]bool{}, reads: map[string]int{}})
	if index, says := s2.Code(context.Background(), ""); index != nil || says != "The latest code, at main, could not be read." {
		t.Errorf("unread latest: %v, %q", index != nil, says)
	}
}

// TestCodeShelfKeepsTheLatestCode never lets the latest code leave for named
// versions, and keeps serving it when it cannot be read again.
func TestCodeShelfKeepsTheLatestCode(t *testing.T) {
	f := &fakeCode{refs: map[string]bool{"main": true, "v3.0.1": true, "v3.0.2": true}, reads: map[string]int{}}
	s, _ := codeShelf(f)
	var clockMu sync.Mutex
	clock := time.Date(2026, 11, 4, 12, 0, 0, 0, time.UTC)
	s.now = func() time.Time { clockMu.Lock(); defer clockMu.Unlock(); return clock }
	s.Code(context.Background(), "")
	s.Code(context.Background(), "v3.0.1")
	s.Code(context.Background(), "v3.0.2")
	s.Code(context.Background(), "")
	if n := f.count("main"); n != 1 {
		t.Errorf("reads of main after two named versions = %d, want 1", n)
	}
	// A read again that fails keeps the code read before, and is asked
	// again after shelfRetry.
	f.mu.Lock()
	delete(f.refs, "main")
	f.mu.Unlock()
	clockMu.Lock()
	clock = clock.Add(codeLatestLifetime)
	clockMu.Unlock()
	s.Code(context.Background(), "")
	deadline := time.Now().Add(5 * time.Second)
	for {
		s.mu.Lock()
		refreshing := s.entries[""].refreshing
		s.mu.Unlock()
		if !refreshing && f.count("main") == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("the read again never ended")
		}
		time.Sleep(time.Millisecond)
	}
	if index, says := s.Code(context.Background(), ""); index == nil || !strings.Contains(says, "read 2026-11-04 12:00 UTC") {
		t.Errorf("Code() after a failed read again = %v, %q", index != nil, says)
	}
	if n := f.count("main"); n != 2 {
		t.Errorf("reads of main within shelfRetry of a failure = %d, want 2", n)
	}
	clockMu.Lock()
	clock = clock.Add(shelfRetry)
	clockMu.Unlock()
	s.Code(context.Background(), "")
	deadline = time.Now().Add(5 * time.Second)
	for f.count("main") != 3 {
		if time.Now().After(deadline) {
			t.Fatal("the latest code was not read again after shelfRetry")
		}
		time.Sleep(time.Millisecond)
	}
}
