package main

import ( log "github.com/Sirupsen/logrus" )
import "sync"
import "net/rpc"
import "net"
import "time"
import "common"
import "net/http"
import "encoding/json"
import "encoding/hex"
import "crypto/sha256"
import "bytes"
import "fmt"

const NODE_URL = "http://6857coin.csail.mit.edu:8080"
const GENESIS_HASH = "169740d5c4711f3cbbde6b9bfbbe8b3d236879d849d1c137660fce9e7884cae7"
const BLOCK_TEXT = "Rolled my own crypto!!!1!!one!!"
const SLEEP_TIME_BETWEEN_SERVER_CALLS_IN_MILLIS = 15000
const SLEEP_TIME_SHORT_IN_MILLIS = 100
const TIMESTAMP_WINDOW_IN_MINUTES = 5
const SEND_THRESHOLD = 1024
const MINE_ON_GENESIS = true

func init() {
  // Only log the warning severity or above.
  log.SetLevel(log.DebugLevel)
}

type Master struct {
	Slaves []Slave
	LastBlock common.Block
	HashChainMap map[uint64][]common.HashChain
	HashChainTriples []common.HashChainTriple
	HashChainTriplesIndex uint64
	NextSlave uint64
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
	master.LastBlock = common.Block{}
	master.LastBlock.ParentId = ""
	master.NextSlave = 0
	rpc.Register(master)
	go listenForSlaves(master)
	
	for {
		b := getNextBlock()
		if MINE_ON_GENESIS {
			b = getGenesisBlock()
		}
		master.mu.Lock()
		if b.ParentId != master.LastBlock.ParentId {
			fmt.Println("Mining on new block")
			master.LastBlock = b
			master.LastBlock.Timestamp = uint64(time.Now().UnixNano())
			shaHash := sha256.Sum256([]byte(BLOCK_TEXT))
			master.LastBlock.Root = hex.EncodeToString(shaHash[:])
			master.HashChainMap = make(map[uint64][]common.HashChain)
			master.HashChainTriples = make([]common.HashChainTriple, 0)
			master.HashChainTriplesIndex = 0
			go master.solveBlock()
		} else if time.Since(master.LastBlock.Timestamp).Minutes() > TIMESTAMP_WINDOW_IN_MINUTES {
			master.LastBlock.Timestamp = uint64(time.Now().UnixNano())
		}
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
	m.mu.Lock()
	fmt.Println("Starting Part A")
	for _, slave := range m.Slaves {
		args := common.HashConfig{m.LastBlock, make([]common.HashChainTriple, 0)}
		reply := false
		slave.conn.Call("Slave.StartStepA", args, &reply)
	}
	m.mu.Unlock()
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

	fmt.Println("getNextBlock returned")

	return data
}

func getGenesisBlock() common.Block {
	res, e := http.Get(NODE_URL + "/block/" + GENESIS_HASH)
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Genesis block failed")
	}
	defer res.Body.Close()

	decoder := json.NewDecoder(res.Body)
	var data common.Block
	e = decoder.Decode(&data)
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Decoder failed")
	}

	fmt.Println("getGenesisBlock returned")

	return data
}

type BlockRequest struct {
	Block common.Block
	Tag string
}

func commitBlock(master *Master, b common.Block, text string) {
	master.mu.Lock()
	br := BlockRequest{b, text}
	s, e := json.Marshal(br)
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Marshalling Failed")
	}
	encoded := string(s)
	fmt.Println(encoded)
	
	resp, e := http.Post(NODE_URL + "/add", "encoding/json", bytes.NewBuffer([]byte(encoded)))
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Marshalling Failed")
	}
	resp.Body.Close()

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
	args := common.HashConfig{m.LastBlock, make([]common.HashChainTriple, 0)}
	bubuj := false
	e = conn.Call("Slave.StartStepA", args, &bubuj)
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Part A error")
	}
	m.mu.Unlock()
	return nil
}

func (m *Master) checkPartADone() bool {
	l := uint64(len(m.HashChainTriples)) - m.HashChainTriplesIndex
	return l >= SEND_THRESHOLD
}

func (m *Master) AddHashChains(chains []common.HashChain, reply *bool) (err error) {
	m.mu.Lock()
	for _, chain := range chains {
		if chain.Timestamp == m.LastBlock.Timestamp {
			fmt.Println("Found hashchain")
			entry, ok := m.HashChainMap[chain.End]
			lengthEndpoint := 1

			if ok {
				duplicate := false
				for _, v := range entry {
					if v.Start == chain.Start {
						duplicate = true
					}
				}

				if !duplicate {
					m.HashChainMap[chain.End] = append(entry, chain)
				}

				lengthEndpoint = len(m.HashChainMap[chain.End])
			} else {
				list := make([]common.HashChain, 0)
				m.HashChainMap[chain.End] = append(list, chain)
			}

			if lengthEndpoint >= 3 {
				entry := m.HashChainMap[chain.End]
				delete(m.HashChainMap, chain.End)

				if entry[0].End < entry[1].End {
					entry[0], entry[1] = entry[1], entry[0]
				}
				if entry[1].End < entry[2].End {
					entry[1], entry[2] = entry[2], entry[1]
				}
				if entry[0].End < entry[1].End {
					entry[0], entry[1] = entry[1], entry[0]
				}

				m.HashChainTriples = append(m.HashChainTriples, common.HashChainTriple{entry[0], entry[1], entry[2]})
				fmt.Println("Appended triple")

				if m.checkPartADone() {
					args := common.HashConfig{m.LastBlock, m.HashChainTriples[m.HashChainTriplesIndex : m.HashChainTriplesIndex + SEND_THRESHOLD]}
					reply := false
					
					done := false
					for !done {
						e := m.Slaves[m.NextSlave].conn.Call("Slave.StartStepB", args, &reply)
						
						if e == nil {
							done = true
						}

						m.NextSlave++
						if m.NextSlave >= uint64(len(m.Slaves)) {
							m.NextSlave = 0
						}
					}


					m.HashChainTriplesIndex += SEND_THRESHOLD
				}
			}
		}
	}
	m.mu.Unlock()

	return nil
}


func (m *Master) SubmitAnswer(triple common.Collision, reply *bool) (err error) {
	m.mu.Lock()
	if triple.Timestamp == m.LastBlock.Timestamp {
		if triple.Nonce1 != triple.Nonce2 && triple.Nonce2 != triple.Nonce3 && triple.Nonce3 != triple.Nonce1 {
			fmt.Println("Found collision")
			b := common.Block{}
			b.ParentId = m.LastBlock.ParentId
			shaHash := sha256.Sum256([]byte(BLOCK_TEXT))
			b.Root = hex.EncodeToString(shaHash[:])
			b.Difficulty = m.LastBlock.Difficulty
			b.Timestamp = m.LastBlock.Timestamp
			b.Nonces[0] = triple.Nonce1
			b.Nonces[1] = triple.Nonce2
			b.Nonces[2] = triple.Nonce3
			b.Version = m.LastBlock.Version
			commitBlock(m, b, BLOCK_TEXT)
		}
	}
	m.mu.Unlock()

	return nil
}