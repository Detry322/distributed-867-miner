package main

import ( log "github.com/Sirupsen/logrus" )
import "sync"
import "net/rpc"
import "net"
import "time"
import "common"
import "net/http"
import "encoding/json"
import "fmt"
import "math"

const NODE_URL = "http://6857coin.csail.mit.edu:8080"
const SLEEP_TIME_BETWEEN_SERVER_CALLS_IN_MILLIS = 15000
const SLEEP_TIME_SHORT_IN_MILLIS = 100

func init() {
  // Only log the warning severity or above.
  log.SetLevel(log.DebugLevel)
}

type Master struct {
	Slaves []Slave
	LastBlock common.Block
	Collisions []Collision
	mu sync.Mutex
}

type Slave struct {
	conn *rpc.Client
}


func main() {
	fmt.Println("Obligatory fmt")
	log.Debug("Entered main")
	master := &Master{}
	master.Slaves = make([]Slave, 0)
	master.LastBlock = getNextBlock()
	master.Collisions = make([]Collisions, 0)
	rpc.Register(master)

	go listenForSlaves(master)
	
	for {
		b := getNextBlock()
		master.mu.Lock()
		if b.Timestamp > master.LastBlock.Timestamp {
			fmt.Println("Mining on new block")
			master.LastBlock = b
			Collisions = make([]Collision, 0)
		}
		go master.solveBlock()
		master.mu.Unlock()
		time.Sleep(time.Millisecond * time.Duration(SLEEP_TIME_BETWEEN_SERVER_CALLS_IN_MILLIS))
	}
}

func listenForSlaves(master *Master) {
	log.Debug("Listening for slaves")
	l, e := net.Listen("tcp", ":1337")
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Listen error")
	}
	rpc.Accept(l)
}

func (m *Master) solveBlock() {
	master.mu.Lock()
	for _, slave := range master.Slaves {
		args := HashConfig{m.LastBlock, make([]Collision)}
		reply := false
		slave.conn.Call("Slave.StartPartA", args, &reply)
	}
	master.mu.Unlock()
}

func getNextBlock() common.Block {
	res, e := http.Get(NODE_URL + "/next")
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Nextblock failed")
	}
	defer res.Body.Close()

	decoder := json.NewDecoder(res.Body)
	var data common.Block
	e = decoder.Decode(&data)
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Decoder failed")
	}

	return data
}

type BlockRequest struct {
	Block common.Block
	Tag string
}

func commitBlock(master *Master, b common.Block) {
	master.mu.Lock()
	req := BlockRequest{b, "Rolled my own crypto"}
	encoded := string(json.Marshal(req))
	fmt.Prinln(encoded)
	req, e := http.NewRequest("POST", NODE_URL + "/add", encoded)
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Creating Request Failed")
	}
	fmt.Prinln(req)

	client := &http.Client{}
	resp, e := client.Do(req)
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Sending Request Failed")
	}
	fmt.Println(resp)

	master.mu.Unlock()
}

func (m *Master) Connect(ip string, reply *bool) (err error) {
	conn, e := rpc.Dial("tcp", ip + ":1337")
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Connect error")
	}
	m.mu.Lock()
	fmt.Println("New slave connected")
	m.Slaves = append(m.Slaves, Slave{conn})
	m.mu.Unlock()
	return nil
}

func (m *Master) partADone() bool {
	l := len(m.Collisions)
	return l*l*l*l > math.Pow(2, m.LastBlock.Difficulty)
}

func (m *Master) AddCollision(collision Collision, reply *bool) (err error) {
	m.mu.Lock()
	if collision.Timestamp >= m.LastBlock.Timestamp {
		m.Collisions = append(m.Collisions, collision)
		if partADone() {
			for _, slave := range master.Slaves {
				args := HashConfig{m.LastBlock, make([]Collision)}
				reply := false
				slave.conn.Call("Slave.StartPartA", args, &reply)
			}
		}
	}
	m.mu.Unlock()
}