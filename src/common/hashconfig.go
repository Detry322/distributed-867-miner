package common

import ()

type HashConfig struct {
	ParentId   []byte
	Root       []byte
	Difficulty int64
	Timestamp  int64
	Version    byte
	Targets    []int64
}
