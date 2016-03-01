package main

import (
	"net/rpc"
	// "common"
	"fmt"
	// "net"
	"sync"
	// "time"
)

const (
	A       = 0
	B       = 1
	address = "18.187.0.60:1337"
)

type Slave struct {
	// Config  common.HashConfig
	Id      int
	Mode    int
	Targets []int64
	mu      sync.Mutex
}

// func (slave *Slave) StartStepA(config common.HashConfig) {
// 	slave.mu.Lock()
// 	slave.Mode = A
// 	// slave.Config = config
// 	slave.Targets = []int64{}
// 	slave.mu.Unlock()
// 	// send stuff to the miner
// }

func (slave *Slave) NewCollisionA(x, y, hash int) {
	// send rpc to master saying there's a new collision
}

func (slave *Slave) NewCollisionB(x, hash int) {
	// send rpc to master saying theres a new b collision
}

// func (slave *Slave) StartStepB(config common.HashConfig, targets []int64) {
// 	slave.Mu.Lock()
// 	defer slave.Mu.Unlock()
// 	// slave.Config = config
// 	slave.Targets = targets
// }

func initiateConnection() {
	client, err := rpc.Dial("tcp", address)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("worked")
	i := 0
	for {
		err = client.Call("Master.Connect", i, &Slave{})
		if err != nil {
			fmt.Println(err)
			return
		}
		i++
	}
	// rpc.Dial(network, address)
}
func main() {
	initiateConnection()
}

func Heartbeat() {
	// make rpc call saying that our id is whatever.
}

/*
This sends a message to the Miner requesting a count for number of hashes computed.
*/
func (slave *Slave) GetCount() int {
	//
	return 0
}
