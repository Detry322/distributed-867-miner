package main

import (
	"common"
	"fmt"
	"log"
	"net/rpc"
	"sync"
	// "time"
)

const (
	A       = 0
	B       = 1
	address = "18.187.0.60:1337"
)

type Slave struct {
	Config common.HashConfig
	Id     int64
	Mode   int64
	mu     sync.Mutex
	Master
}

func (slave *Slave) StartStepA(config common.HashConfig) {
	slave.mu.Lock()
	slave.Mode = A
	slave.Config = config
	slave.mu.Unlock()
	// send stuff to the miner
}

func (slave *Slave) NewCollisionA(x, y, hash int64) {
	// send rpc to master saying there's a new collision
}

func (slave *Slave) NewCollisionB(x, hash int64) {
	// send rpc to master saying theres a new b collision

}

/*
This is used when we want this slave to transition from
*/
func (slave *Slave) StartStepB(config common.HashConfig, reply *bool) (err error) {
	slave.mu.Lock()
	defer slave.mu.Unlock()
	slave.Config = config
	// TODO: send message to miner starting step b
}

func getIPAddress() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		log.Fatalln(err)
	}

	for _, a := range addrs {
		if ipnet, ok := a.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				os.Stdout.WriteString(ipnet.IP.String() + "\n")
			}
		}
	}
}

func listenToMiner() {
	for {
		//listen to messages.
	}
}

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

/*
Runs on startup
*/
func main() {
	initiateConnection()
}

/*
RPC method called when master wants to ensure connection is still live.
*/
func (slave *Slave) Heartbeat(args int, reply *int) (err error) {
	*reply = slave.GetCount()
}

/*
This sends a message to the Miner requesting a count for number of hashes computed.
*/
func (slave *Slave) GetCount() int {
	//
	return 0
}
