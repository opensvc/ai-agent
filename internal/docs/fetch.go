package docs

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
)

const (
	// maxIndexBytes bounds llms.txt, maxPageBytes each page it lists, and
	// maxPages and maxBookBytes the pages read in all.
	maxIndexBytes = 1 << 20
	maxPageBytes  = 1 << 20
	maxPages      = 1024
	maxBookBytes  = 16 << 20
	// fetchers is the number of pages read at once.
	fetchers = 8
)

var (
	indexSection = regexp.MustCompile(`^## (.+?)\s*$`)
	indexLink    = regexp.MustCompile(`^- \[[^\]]*\]\(([^)\s]+\.md)\)`)
	indexBuild   = regexp.MustCompile(`^Build: (.+?)\.?\s*$`)
	indexRelease = regexp.MustCompile(`^Keyword reference: (.+?)\.?\s*$`)
)

// Book is the documentation an llms.txt index lists: its pages, the build it
// says it is and the agent release its keyword reference documents, each
// empty when it says none.
type Book struct {
	Pages            []Page
	Build            string
	KeywordReference string
}

// Fetch reads the llms.txt index at index, then each page it lists that
// include accepts, given its section and its path relative to the index: the
// section of the pages a reader may skip, after https://llmstxt.org, is
// Optional, and include picks there too. A page is read from the host of the index
// only, a redirect elsewhere refused. Any failure fails the whole, the reads
// left cancelled: a book read in part answers in part.
func Fetch(ctx context.Context, client *http.Client, index string, include func(section, path string) bool) (Book, error) {
	base, err := url.Parse(index)
	if err != nil || base.Scheme != "https" || base.Host == "" {
		return Book{}, fmt.Errorf("documentation index %q is not an https URL", index)
	}
	sameHost := *client
	sameHost.CheckRedirect = func(request *http.Request, via []*http.Request) error {
		if request.URL.Scheme != base.Scheme || request.URL.Host != base.Host {
			return fmt.Errorf("redirect to %s refused: the documentation is read from %s only", request.URL.Host, base.Host)
		}
		if len(via) >= 10 {
			return fmt.Errorf("stopped after 10 redirects")
		}
		return nil
	}
	client = &sameHost
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	text, err := get(ctx, client, base.String(), maxIndexBytes)
	if err != nil {
		return Book{}, err
	}
	var (
		book    Book
		links   []string
		section string
	)
	// The index is read whole, bounded by maxIndexBytes: its lines are cut
	// in memory, a line of any length kept, rather than through a scanner
	// that stops on a line longer than its buffer and drops the pages after.
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSuffix(line, "\r")
		if match := indexBuild.FindStringSubmatch(line); match != nil && book.Build == "" {
			book.Build = match[1]
		}
		if match := indexRelease.FindStringSubmatch(line); match != nil && book.KeywordReference == "" {
			book.KeywordReference = match[1]
		}
		if match := indexSection.FindStringSubmatch(line); match != nil {
			section = match[1]
			continue
		}
		match := indexLink.FindStringSubmatch(line)
		if match == nil {
			continue
		}
		link, err := base.Parse(match[1])
		if err != nil || link.Host != base.Host || link.Scheme != base.Scheme {
			continue
		}
		path := strings.TrimPrefix(link.Path, strings.TrimSuffix(base.Path, "llms.txt"))
		if include(section, path) {
			links = append(links, link.String())
			book.Pages = append(book.Pages, Page{Path: path})
		}
	}
	if len(links) == 0 {
		return Book{}, fmt.Errorf("documentation index %s lists no page to read", index)
	}
	if len(links) > maxPages {
		return Book{}, fmt.Errorf("documentation index %s lists %d pages, more than %d", index, len(links), maxPages)
	}
	var (
		wait    sync.WaitGroup
		mu      sync.Mutex
		total   int
		failure error
		slots   = make(chan struct{}, fetchers)
	)
	for i, link := range links {
		wait.Add(1)
		slots <- struct{}{}
		go func() {
			defer wait.Done()
			defer func() { <-slots }()
			page, err := get(ctx, client, link, maxPageBytes)
			mu.Lock()
			defer mu.Unlock()
			if total += len(page); err == nil && total > maxBookBytes {
				err = fmt.Errorf("documentation exceeds %d bytes", maxBookBytes)
			}
			if err != nil && failure == nil {
				failure = err
				cancel()
			}
			book.Pages[i].Text = page
		}()
	}
	wait.Wait()
	if failure != nil {
		return Book{}, failure
	}
	return book, nil
}

// get reads a text of at most limit bytes.
func get(ctx context.Context, client *http.Client, link string, limit int64) (string, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, link, nil)
	if err != nil {
		return "", err
	}
	response, err := client.Do(request)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", link, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return "", &statusError{link: link, code: response.StatusCode, status: response.Status}
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return "", fmt.Errorf("read %s: %w", link, err)
	}
	if int64(len(data)) > limit {
		return "", fmt.Errorf("read %s: more than %d bytes", link, limit)
	}
	return string(data), nil
}

// statusError is a read answered with a status other than 200 OK.
type statusError struct {
	link   string
	code   int
	status string
}

func (e *statusError) Error() string { return fmt.Sprintf("read %s: %s", e.link, e.status) }
