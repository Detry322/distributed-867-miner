package common

import (
	"encoding/binary"
	"encoding/hex"
)

type HashConfig struct {
	Block   Block
	Triples []HashChainTriple
}

type HashChain struct {
	Start     uint64
	End       uint64
	Length    uint64
	Timestamp uint64
}

type HashChainTriple struct {
	Chain1 HashChain // Longest chain
	Chain2 HashChain // Middle chain
	Chain3 HashChain // Shortest chain
}

type Collision struct {
	Nonce1    uint64
	Nonce2    uint64
	Nonce3    uint64
	Timestamp uint64
}

type Block struct {
	ParentId   string
	Root       string
	Difficulty uint64
	Timestamp  uint64
	Nonces     [3]uint64
	Version    byte
}

func AddHexDigits(message string, newInt uint64, want16 bool) string {
	if want16 {
		// make it 16 bytes
		array := make([]byte, 8)
		binary.BigEndian.PutUint64(array, newInt)
		message = message + hex.EncodeToString(array)
	} else {
		// make it 8 bytes
		array := make([]byte, 4)
		binary.BigEndian.PutUint32(array, uint32(newInt))
		message = message + hex.EncodeToString(array)
	}
	return message
}
