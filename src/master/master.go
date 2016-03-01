package main

import ( log "github.com/Sirupsen/logrus" )
import "sync"
import "net/rpc"
import "net"
import "time"
//import "fmt"

const NODE_URL = "http://6857coin.csail.mit.edu:8080"
const SLEEP_TIME_BETWEEN_SERVER_CALLS_IN_MILLIS = 15000

func init() {
  // Only log the warning severity or above.
  log.SetLevel(log.DebugLevel)
}

type Master struct {
	Slaves []Slave

	mu sync.Mutex
}


type Slave struct {
	conn *rpc.Client
	MatchIndex int
}

type Block struct {

}

type HashResult struct {
	PreImage int
	Hash int
}

func Make() *Master {
	master := &Master{}
	master.Slaves = make([]Slave, 0)



	return master
}

func main() {
	log.Debug("Entered main")
	master := &Master{}
	master.Slaves = make([]Slave, 0)
	rpc.Register(master)

	go listenForSlaves(master)
	for {time.Sleep(time.Millisecond)}
}

func listenForSlaves(master *Master) {
	log.Debug("Listening for slaves")
	l, e := net.Listen("tcp", ":1337")
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Fatal("Listen error")
	}
	rpc.Accept(l)
}

func solveBlock(b Block) {

}

func getNextBlock() Block {

}

func commitBlock(b Block) {

}

func (m *Master) Connect(ip string, reply *bool) (err error) {
	conn, e := rpc.Dial("tcp", ip + ":1337")
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Fatal("Connect error")
	}
	m.Slaves = append(m.Slaves, Slave{conn, -1})
	return nil
}