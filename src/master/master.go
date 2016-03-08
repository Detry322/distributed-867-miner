package main

import (
	log "github.com/Sirupsen/logrus"
)
import "sync"
import "strconv"
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

const NANOS_PER_MINUTE = 1000 * 1000 * 1000 * 60
const NODE_URL = "http://6857coin.csail.mit.edu:8080"
const BLOCK_TEXT = "e4a18a0032ca05abe9c6e2d69ab7cea013da79101fbf942707ae2778ed770199"
const SLEEP_TIME_BETWEEN_SERVER_CALLS_IN_MILLIS = 5000
const SLEEP_TIME_SHORT_IN_MILLIS = 100
const TIMESTAMP_WINDOW_IN_MINUTES = 8
const SEND_THRESHOLD = 2048

func init() {
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

type Master struct {
	Slaves                []Slave
	LastBlock             common.Block
	HashChainMap          map[uint64][]common.HashChain
	HashChainTriples      []common.HashChainTriple
	HashChainTriplesIndex uint64
	NextSlave             uint64
	mu                    sync.Mutex
}

type Slave struct {
	conn *rpc.Client
}

func main() {
	fmt.Println("Started")
	log.Debug("Entered main")
	master := &Master{}
	master.Slaves = make([]Slave, 0)
	master.LastBlock = common.Block{}
	master.LastBlock.ParentId = ""
	master.NextSlave = 0
	rpc.Register(master)
	go listenForSlaves(master)

	for {
		log.Debug("Started main for loop")
		b := getNextBlock()
		log.Debug("Waiting for lock in main for loop")
		master.mu.Lock()
		log.Debug("Acquired lock in main for loop")
		if b.ParentId != master.LastBlock.ParentId || (int64(time.Now().UnixNano())-int64(master.LastBlock.Timestamp))/NANOS_PER_MINUTE > TIMESTAMP_WINDOW_IN_MINUTES {
			fmt.Println("Mining on new block")
			master.LastBlock = b
			master.LastBlock.Timestamp = uint64(time.Now().Add(time.Minute * time.Duration(TIMESTAMP_WINDOW_IN_MINUTES)).UnixNano())
			shaHash := sha256.Sum256([]byte(BLOCK_TEXT))
			master.LastBlock.Root = hex.EncodeToString(shaHash[:])
			master.HashChainMap = make(map[uint64][]common.HashChain)
			master.HashChainTriples = make([]common.HashChainTriple, 0)
			master.HashChainTriplesIndex = 0
			go master.solveBlock()
		}
		log.Debug("Releasing lock in main for loop")
		master.mu.Unlock()
		log.Debug("Sleeping in main for loop")
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
	log.Debug("Entered solveBlock")
	log.Debug("Waiting for lock in solveBlock")
	m.mu.Lock()
	log.Debug("Acquired lock in solveBlock")
	for _, slave := range m.Slaves {
		args := common.HashConfig{m.LastBlock, make([]common.HashChainTriple, 0)}
		reply := false
		slave.conn.Call("Slave.StartStepA", args, &reply)
	}
	log.Debug("Releasing lock in solveBlock")
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
	log.Debug("Returning in getNextBlock")

	return data
}

type BlockRequest struct {
	header BlockHeader
	block   string
}

type BlockHeader struct {
	parentId string
	root string
	difficulty uint64
	timestamp uint64
	nonces [3]uint64
	version byte
}

func commitBlock(master *Master, b common.Block, text string) {
	fmt.Println("parent id", b.ParentId)
	fmt.Println("root", b.Root)
	fmt.Println("diff", b.Difficulty)
	fmt.Println("ts", b.Timestamp)
	fmt.Println("1", b.Nonces[0])
	fmt.Println("2", b.Nonces[1])
	fmt.Println("3", b.Nonces[2])
	fmt.Println("ver", b.Version)
	fmt.Println(text)
	log.Info("Attempting to commit block!!!!!!!")
	encoded := fmt.Sprintf(`
{
  "header": {
    "parentid": "%s",
    "root": "%s",
    "difficulty": %d,
    "timestamp": %d,
    "nonces": [
      %d,
      %d,
      %d
    ],
    "version": %d
  },
  "block": "%s"
}
`, b.ParentId, b.Root, b.Difficulty, b.Timestamp, b.Nonces[0], b.Nonces[1], b.Nonces[2], b.Version, text)
	fmt.Println(encoded)

	resp, e := http.Post(NODE_URL+"/add", "encoding/json", bytes.NewBuffer([]byte(encoded)))
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Marshalling Failed")
	}
	resp.Body.Close()
	log.Warn(resp.Status)
	log.Error("Attempted to commit block!!!!!!!")
	getNextBlock()
}

func (m *Master) Connect(ip string, reply *bool) (err error) {
	conn, e := rpc.Dial("tcp", ip+":1336")
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Connect error")
	}
	log.Debug("Waiting for lock in Connect")
	m.mu.Lock()
	log.Debug("Acquired for lock in Connect")
	fmt.Println("New slave connected")
	m.Slaves = append(m.Slaves, Slave{conn})
	args := common.HashConfig{m.LastBlock, make([]common.HashChainTriple, 0)}
	bubuj := false
	e = conn.Call("Slave.StartStepA", args, &bubuj)
	if e != nil {
		log.WithFields(log.Fields{"error": e}).Error("Part A error")
	}
	log.Debug("Releasing lock in Connect")
	m.mu.Unlock()
	return nil
}

func (m *Master) checkPartADone() bool {
	l := uint64(len(m.HashChainTriples)) - m.HashChainTriplesIndex
	return l >= SEND_THRESHOLD
}

func (m *Master) AddHashChains(chains []common.HashChain, reply *bool) (err error) {
	log.Debug("Waiting for lock in AddHashChains")
	m.mu.Lock()
	log.Debug("Waiting for lock in AddHashChains")
	for _, chain := range chains {
		if chain.Timestamp == m.LastBlock.Timestamp {
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

				if entry[0].Length < entry[1].Length {
					entry[0], entry[1] = entry[1], entry[0]
				}
				if entry[1].Length < entry[2].Length {
					entry[1], entry[2] = entry[2], entry[1]
				}
				if entry[0].Length < entry[1].Length {
					entry[0], entry[1] = entry[1], entry[0]
				}

				m.HashChainTriples = append(m.HashChainTriples, common.HashChainTriple{entry[0], entry[1], entry[2]})
				if len(m.HashChainTriples)%1000 == 0 {
					fmt.Println(chain.Length)
					log.Info("triples", strconv.Itoa(len(m.HashChainTriples)))
				}
				if m.checkPartADone() {

					go sendPartBRequest(m, m.HashChainTriplesIndex, m.LastBlock.Timestamp)

					m.HashChainTriplesIndex += SEND_THRESHOLD
				}
			}
		}
	}
	log.Debug("Releasing lock in AddHashChains")
	m.mu.Unlock()

	return nil
}

func sendPartBRequest(m *Master, idx uint64, timestamp uint64) {
	if m.LastBlock.Timestamp != timestamp {
		log.Debug("1")
		return
	}
	args := common.HashConfig{m.LastBlock, m.HashChainTriples[idx : idx+SEND_THRESHOLD]}
	reply := false
	log.Debug("2")
	done := false
	for !done {
		log.Debug("Waiting for lock in sendPartBRequest")
		m.mu.Lock()
		log.Debug("Acquired lock in sendPartBRequest")
		m.NextSlave++
		if m.NextSlave >= uint64(len(m.Slaves)) {
			m.NextSlave = 0
		}
		slaveNum := m.NextSlave
                m.mu.Unlock()
		log.Debug("Releasing lock in sendPartBRequest")
		e := m.Slaves[slaveNum].conn.Call("Slave.StartStepB", args, &reply)
		if e == nil {
			done = true
		}
	}
}

func (m *Master) SubmitAnswer(triple common.Collision, reply *bool) (err error) {
	log.Debug("Waiting for lock in SubmitAnswer")
	m.mu.Lock()
	log.Debug("Acquired for lock in SubmitAnswer")
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
	log.Debug("Releasing lock in SubmitAnswer")
	m.mu.Unlock()

	return nil
}
