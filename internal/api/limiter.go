package api

import "sync"

// askLimiter bounds the runs of the agent: maxConcurrent in the whole
// process, every user, backend and cluster together, and perUser for one
// user of one backend and cluster, so that one user cannot hold every slot
// and deny the runs of the others.
type askLimiter struct {
	mu            sync.Mutex
	maxConcurrent int
	perUser       int
	running       int
	byUser        map[string]int
}

// limiterRefusal tells which bound refused a run.
type limiterRefusal int

const (
	limiterAccepted limiterRefusal = iota
	limiterFull
	limiterUserFull
)

func newAskLimiter(maxConcurrent, perUser int) *askLimiter {
	return &askLimiter{maxConcurrent: maxConcurrent, perUser: perUser, byUser: map[string]int{}}
}

// userKey is the key of a user: its backend, its cluster, its issuer and
// its subject. The cluster is the tenant: a local user of om3 names its
// node as its issuer, and two clusters may have a node of the same name,
// and a root user each.
func userKey(backend, clusterID, issuer, subject string) string {
	return backend + "\x00" + clusterID + "\x00" + issuer + "\x00" + subject
}

// tryAcquire takes a slot for the user of key: the instance full refuses
// first, then the share of the user, which is told only while others may
// still run.
func (l *askLimiter) tryAcquire(key string) limiterRefusal {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.running >= l.maxConcurrent {
		return limiterFull
	}
	if l.byUser[key] >= l.perUser {
		return limiterUserFull
	}
	l.running++
	l.byUser[key]++
	return limiterAccepted
}

// release gives back the slot of the user of key.
func (l *askLimiter) release(key string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.running--
	if l.byUser[key]--; l.byUser[key] <= 0 {
		delete(l.byUser, key)
	}
}
