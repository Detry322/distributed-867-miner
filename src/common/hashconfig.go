package common

import ()

type HashConfig struct {
	ParentId   []byte
	Root       []byte
	Difficulty int64
	Timestamp  uint64
	Version    byte
	Targets    []int64 // hashes that we want to match, if in mode B
}

type CollisionA struct {
	X, Y      int64
	Timestamp uint64
}

type CollisionB struct {
	X, Hash   int64
	Timestamp uint64
}
