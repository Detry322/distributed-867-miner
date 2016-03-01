package common

import (
	"net/rpc"
	"sync"
)

type HashConfig struct {
	ParentId   int
	NameHash   int
	Difficulty int
	Timestamp  int
}
