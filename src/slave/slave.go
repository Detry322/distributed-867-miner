package main

import (
	"common"
	"errors"
	"fmt"
	"net"
	"net/rpc"
	"sync"
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
	Master *rpc.Client
}

func (slave *Slave) StartStepA(config common.HashConfig, reply *bool) (err error) {
	slave.mu.Lock()
	defer slave.mu.Unlock()
	if config.Timestamp < slave.Config.Timestamp {
		*reply = false
		return
	}
	slave.Mode = A
	slave.Config = config
	// send stuff to the miner
	return nil
}

func (slave *Slave) NewCollisionA(x, y int64) (err error) {
	// send rpc to master saying there's a new collision
	collision := common.CollisionB{X: x, Y: y, Timestamp: slave.Config.Timestamp}
	// send collision to sidd.
}

func (slave *Slave) NewCollisionB(x, hash int64) {
	// send rpc to master saying theres a new b collision
	collision := common.CollisionB{X: x, Hash: hash, Timestamp: slave.Config.Timestamp}
	// send collision to sidd.
}

/*
This is used when we want this slave to transition from
*/
func (slave *Slave) StartStepB(config common.HashConfig, reply *bool) (err error) {
	slave.mu.Lock()
	defer slave.mu.Unlock()
	slave.Config = config
	if config.Timestamp < slave.Config.Timestamp {
		*reply = false
		return
	}
	slave.Config = config
	slave.Mode = B
	// TODO: send message to miner starting step b
	return nil
}

func getIPAddress() (s string, err error) {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "", err
	}
	for _, a := range addrs {
		if ipnet, ok := a.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				fmt.Println(ipnet.IP.String())
				return ipnet.IP.String(), nil
			}
		}
	}
	return "", errors.New("no ip found")
}

func (slave *Slave) listenToMiner() {
	for {
		//listen to messages.
	}
}

func initiateConnection(slave *Slave) {
	client, err := rpc.Dial("tcp", address)
	if err != nil {
		fmt.Println(err)
		return
	}
	slave.Master = client
	fmt.Println("established connection")
	ip, err := getIPAddress()
	if err != nil {
		// ip addr is fucked.

	}
	reply := false
	err = slave.Master.Call("Master.Connect", ip, &reply)
	if err != nil {
		fmt.Println(err)
		return
	}
	slave.listenToMiner()
}

/*
Runs on startup
*/
func main() {
	slave := &Slave{}
	go listenForMaster(slave)
	initiateConnection(slave)
}

func listenForMaster(slave *Slave) {
	// log.Debug("Listening for slaves")
	l, e := net.Listen("tcp", ":1337")
	if e != nil {
		fmt.Println("fuck!")
		// log.WithFields(log.Fields{"error": e}).Fatal("Listen error")
	}
	rpc.Accept(l)
}

/*
RPC method called when master wants to ensure connection is still live.
*/
func (slave *Slave) Heartbeat(args int, reply *int64) (err error) {
	*reply = slave.GetCount()
	return nil
}

/*
This sends a message to the Miner requesting a count for number of hashes computed.
*/
func (slave *Slave) GetCount() int64 {
	// TODO: should send message.
	return 0
}
