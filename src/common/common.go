package common

import ()

type HashConfig struct {
	Block Block
	Targets []Collision
}

type Collision struct {
	X uint64
	Y uint64
	Hash uint64
	Timestamp uint64
}

type Block struct {
	ParentId string
	Root string
	Difficulty uint64
	Timestamp uint64
	Nonces [3]uint64
	Version byte
}