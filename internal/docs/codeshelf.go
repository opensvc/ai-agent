package docs

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"
)

const (
	// maxCodeIndexes bounds the code indexes kept in memory, one per version
	// of the code: the latest code, and the versions the clusters of the
	// conversations run, an index weighing tens of MB.
	maxCodeIndexes = 3
	// codeLoadTimeout bounds the reading and the indexing of the code of a
	// version, the search waiting on it.
	codeLoadTimeout = 2 * time.Minute
	// codeLatestLifetime is how long the latest code, a branch that moves,
	// is kept before it is read again; a tag or a commit is kept as long as
	// it is used.
	codeLatestLifetime = 6 * time.Hour
	// maxCodeLoads bounds the reads of code at once, each downloading and
	// indexing an archive: the others wait their turn.
	maxCodeLoads = 2
)

// CodeShelf holds the code indexes of the versions of a program searched,
// read from the archive of each version the first time a search asks for
// it: the least recently used one leaves past maxCodeIndexes, and a read that
// failed is asked again after shelfRetry.
type CodeShelf struct {
	client   *http.Client
	template string
	// latest is the reference of the latest code, such as main, read when
	// a search names no version.
	latest string
	fetch  func(ctx context.Context, ref string) ([]File, error)
	logf   func(format string, args ...any)
	now    func() time.Time

	// loads holds a token per read of code in progress, maxCodeLoads.
	loads chan struct{}

	mu      sync.Mutex
	entries map[string]*codeEntry
	used    uint64
}

// codeEntry is the code of one version, read once: done closes when the read
// ends. refreshing tells that the latest code, past its lifetime, is read
// again, the entry served meanwhile. waiters are the searches waiting on
// the read, and cancel stops it: the last to leave stops a read nobody waits
// on any more.
type codeEntry struct {
	done       chan struct{}
	waiters    int
	cancel     context.CancelFunc
	index      *CodeIndex
	note       string
	err        error
	at         time.Time
	used       uint64
	refreshing bool
}

// NewCodeShelf makes the shelf of the code of the archives served at
// template, such as https://codeload.github.com/opensvc/om3/tar.gz/{ref},
// the latest code at the reference latest, such as main; logf tells each
// read, what it read and why it failed.
func NewCodeShelf(client *http.Client, template, latest string, logf func(format string, args ...any)) *CodeShelf {
	if logf == nil {
		logf = func(string, ...any) {}
	}
	s := &CodeShelf{client: client, template: template, latest: latest, logf: logf, now: time.Now, entries: map[string]*codeEntry{}, loads: make(chan struct{}, maxCodeLoads)}
	s.fetch = func(ctx context.Context, ref string) ([]File, error) {
		return FetchCode(ctx, s.client, s.template, ref)
	}
	return s
}

// Latest is the reference of the latest code.
func (s *CodeShelf) Latest() string { return s.latest }

// Code gives the code index of version, such as v3.0.0-rc56 or
// v3.0.0-rc56-41-g451250d44, the latest code when version is empty, and what
// to tell of it: the code of that commit, else of the tag it is past when the
// commit is not published, said not to be exactly that version; nil, saying
// why, when version names no code or it cannot be read.
func (s *CodeShelf) Code(ctx context.Context, version string) (*CodeIndex, string) {
	if version != "" {
		version = withV(version)
		if ref, _ := CodeRef(version); ref == "" {
			return nil, fmt.Sprintf("%q names no version of the code: name a tag in full, such as v<major>.<minor>.<patch>-rc<n>, or a commit, or leave the version out for the latest code.", version)
		}
	}
	entry := s.entry(version)
	if entry == nil {
		return nil, fmt.Sprintf("The code of %s cannot be read now: %d versions of the code are kept or being read, the most at once; search one of them, or the latest code, or try again in a few minutes.", label(version), maxCodeIndexes)
	}
	select {
	case <-entry.done:
	case <-ctx.Done():
		s.leave(version, entry)
		return nil, fmt.Sprintf("The code of %s could not be read: %v.", label(version), ctx.Err())
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	entry.waiters--
	if entry.err != nil {
		if version != "" && errors.Is(entry.err, errNotPublished) {
			return nil, fmt.Sprintf("No code is published at %s: name a tag in full, such as v<major>.<minor>.<patch>-rc<n>, or a commit.", version)
		}
		if version == "" {
			return nil, fmt.Sprintf("The latest code, at %s, could not be read.", s.latest)
		}
		return nil, fmt.Sprintf("The code of %s could not be read.", version)
	}
	return entry.index, entry.note
}

// entry gives the entry of version, the caller counted among its waiters,
// read in the background the first time, and again after shelfRetry when it
// failed. A new read takes the place of the least recently used entry read,
// never of one in progress, nor of the latest code, which any search naming
// no version reads; nil when every place is taken: the reads in progress are
// bounded with the indexes kept.
func (s *CodeShelf) entry(version string) *codeEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.used++
	entry, ok := s.entries[version]
	if ok && entry.err != nil && s.now().Sub(entry.at) >= shelfRetry {
		delete(s.entries, version)
		ok = false
	}
	if ok {
		entry.used = s.used
		entry.waiters++
		// The latest code moves: past its lifetime it is read again, the
		// searches served the code read before until the new read ends.
		if version == "" && entry.ended() && entry.err == nil && !entry.refreshing && s.now().Sub(entry.at) >= codeLatestLifetime {
			entry.refreshing = true
			go s.refresh(entry)
		}
		return entry
	}
	for len(s.entries) >= maxCodeIndexes {
		oldest := ""
		for v, e := range s.entries {
			if v == "" || !e.ended() {
				continue
			}
			if oldest == "" || e.used < s.entries[oldest].used {
				oldest = v
			}
		}
		if oldest == "" {
			return nil
		}
		delete(s.entries, oldest)
	}
	ctx, cancel := context.WithTimeout(context.Background(), codeLoadTimeout)
	entry = &codeEntry{done: make(chan struct{}), used: s.used, waiters: 1, cancel: cancel}
	s.entries[version] = entry
	go func() {
		defer cancel()
		index, note, err := s.load(ctx, version)
		s.mu.Lock()
		entry.index, entry.note, entry.err, entry.at = index, note, err, s.now()
		s.mu.Unlock()
		close(entry.done)
	}()
	return entry
}

// leave takes a search that stopped waiting out of the waiters of entry: the
// last one stops the read, and the entry leaves the shelf, read again by the
// next search that asks for it.
func (s *CodeShelf) leave(version string, entry *codeEntry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if entry.waiters--; entry.waiters > 0 || entry.ended() {
		return
	}
	entry.cancel()
	if s.entries[version] == entry {
		delete(s.entries, version)
	}
}

// load reads and indexes the code of version once a place among the
// maxCodeLoads reads at once is free, within ctx.
func (s *CodeShelf) load(ctx context.Context, version string) (*CodeIndex, string, error) {
	select {
	case s.loads <- struct{}{}:
		defer func() { <-s.loads }()
	case <-ctx.Done():
		return nil, "", ctx.Err()
	}
	return s.read(ctx, version)
}

// refresh reads the latest code again, then puts it in place of old; a read
// that fails keeps old, read again after shelfRetry.
func (s *CodeShelf) refresh(old *codeEntry) {
	ctx, cancel := context.WithTimeout(context.Background(), codeLoadTimeout)
	defer cancel()
	index, note, err := s.load(ctx, "")
	s.mu.Lock()
	defer s.mu.Unlock()
	old.refreshing = false
	if err != nil {
		old.at = s.now().Add(shelfRetry - codeLatestLifetime)
		return
	}
	fresh := &codeEntry{done: make(chan struct{}), index: index, note: note, at: s.now(), used: old.used}
	close(fresh.done)
	if s.entries[""] == old {
		s.entries[""] = fresh
	}
}

// ended tells whether the read of an entry ended.
func (e *codeEntry) ended() bool {
	select {
	case <-e.done:
		return true
	default:
		return false
	}
}

// read reads and indexes the code of version: its commit, else its tag,
// said in the log. A panic of the indexing, on a file it did not foresee, is
// an error.
func (s *CodeShelf) read(ctx context.Context, version string) (index *CodeIndex, note string, err error) {
	began := s.now()
	ref, tag := CodeRef(version)
	if version == "" {
		ref = s.latest
	}
	var files []File
	defer func() {
		if r := recover(); r != nil {
			index, note, err = nil, "", fmt.Errorf("index the code of %s: panic: %v", version, r)
		}
		if err != nil {
			s.logf("code: read %s at %s: %v", label(version), ref, err)
			return
		}
		s.logf("code: %s read at %s, %d files, %d declarations, %d files left out, in %s", label(version), ref, len(files), index.Len(), index.Skipped, s.now().Sub(began).Round(time.Millisecond))
	}()
	note = fmt.Sprintf("This is the code of %s.", version)
	if version == "" {
		note = fmt.Sprintf("This is the latest code of om3, at %s, read %s: a cluster may run another version.", ref, began.UTC().Format("2006-01-02 15:04 UTC"))
	}
	files, err = s.fetch(ctx, ref)
	if errors.Is(err, errNotPublished) && tag != "" {
		s.logf("code: %s is not published at %s: its tag %s is read", version, ref, tag)
		note = fmt.Sprintf("The code of %s is not published: this is the code of its tag %s, which is not exactly the code that runs.", version, tag)
		ref = tag
		files, err = s.fetch(ctx, ref)
	}
	if err != nil {
		return nil, "", err
	}
	index, err = NewCode(files, "the code of "+ref)
	return index, note, err
}

// withV gives a tag, or a version past a tag, the v the tags carry, such as
// v3.0.0-rc56 for 3.0.0-rc56; a commit as it is.
func withV(version string) string {
	if version[0] >= '0' && version[0] <= '9' && !commitVersion.MatchString(version) && (tagVersion.MatchString(version) || describeVersion.MatchString(version)) {
		return "v" + version
	}
	return version
}

// label names a version in the log, the latest code included.
func label(version string) string {
	if version == "" {
		return "the latest code"
	}
	return version
}
