package common

import ()

type HashConfig struct {
	Block Block
	Targets []Collision
}

type HashChain struct {
	Start uint64
	End uint64
	Length uint64
	Timestamp uint64
}

type HashChainTriple struct {
	Chain1 HashChain // Longest chain
	Chain2 HashChain // Middle chain
	Chain3 HashChain // Shortest chain
}

type Block struct {
	ParentId string
	Root string
	Difficulty uint64
	Timestamp uint64
	Nonces [3]uint64
	Version byte
}