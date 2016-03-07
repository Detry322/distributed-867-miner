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
const GENESIS_HASH = "cd622d4f86b820611dc776fe23cd76a07aad183d1d1b33f504a3940e76da28f3"
var BLOCK_TEXTS = [...]string{"00001                   Presented by Rolled my own crypto!!!1!!one!!", "00002", "00003                                     _______", "00004                              _,,ad8888888888bba,_", "00005                           ,ad88888I888888888888888ba,", "00006                         ,88888888I88888888888888888888a,", "00007                       ,d888888888I8888888888888888888888b,", "00008                      d88888PP```` ``YY88888888888888888888b,", "00009                    ,d88``__,,--------,,,,.;ZZZY8888888888888,", "00010                   ,8IIl``                ;;l`ZZZIII8888888888,", "00011                  ,I88l;`                  ;lZZZZZ888III8888888,", "00012                ,II88Zl;.                  ;llZZZZZ888888I888888,", "00013               ,II888Zl;.                .;;;;;lllZZZ888888I8888b", "00014              ,II8888Z;;                 `;;;;;`llZZ8888888I8888,", "00015              II88888Z;`                        .;lZZZ8888888I888b", "00016              II88888Z; _,aaa,      .,aaaaa,__.l;llZZZ88888888I888", "00017              II88888IZZZZZZZZZ,  .ZZZZZZZZZZZZZZ;llZZ88888888I888,", "00018              II88888IZZ<`(@@>Z|  |ZZZ<`(@@>ZZZZ;;llZZ888888888I88I", "00019             ,II88888;   ```` ;|  |ZZ; ````     ;;llZ8888888888I888", "00020             II888888l            `;;          .;llZZ8888888888I888,", "00021            ,II888888Z;           ;;;        .;;llZZZ8888888888I888I", "00022            III888888Zl;    ..,   `;;       ,;;lllZZZ88888888888I888", "00023            II88888888Z;;...;(_    _)      ,;;;llZZZZ88888888888I888,", "00024            II88888888Zl;;;;;` `--`Z;.   .,;;;;llZZZZ88888888888I888b", "00025            ]I888888888Z;;;;`   `;llllll;..;;;lllZZZZ88888888888I8888,", "00026            II888888888Zl.;;`Y88bd888P`;;,..;lllZZZZZ88888888888I8888I", "00027            II8888888888Zl;.; ``PPP`;;;,..;lllZZZZZZZ88888888888I88888", "00028            II888888888888Zl;;. `;;;l;;;;lllZZZZZZZZW88888888888I88888", "00029            `II8888888888888Zl;.    ,;;lllZZZZZZZZWMZ88888888888I88888", "00030             II8888888888888888ZbaalllZZZZZZZZZWWMZZZ8888888888I888888,", "00031             `II88888888888888888b`WWZZZZZWWWMMZZZZZZI888888888I888888b", "00032              `II88888888888888888;ZZMMMMMMZZZZZZZZllI888888888I8888888", "00033               `II8888888888888888 `;lZZZZZZZZZZZlllll888888888I8888888,", "00034                II8888888888888888, `;lllZZZZllllll;;.Y88888888I8888888b,", "00035               ,II8888888888888888b   .;;lllllll;;;.;..88888888I88888888b,", "00036               II888888888888888PZI;.  .`;;;.;;;..; ...88888888I8888888888,", "00037               II888888888888PZ;;`;;.   ;. .;.  .;. .. Y8888888I88888888888b,", "00038              ,II888888888PZ;;`                        `8888888I8888888888888b,", "00039              II888888888`                              888888I8888888888888888b", "00040             ,II888888888                              ,888888I88888888888888888", "00041            ,d88888888888                              d888888I8888888888ZZZZZZZ", "00042         ,ad888888888888I                              8888888I8888ZZZZZZZZZZZZZ", "00043       ,d888888888888888`                              888888IZZZZZZZZZZZZZZZZZZ", "00044     ,d888888888888P`8P`                               Y888ZZZZZZZZZZZZZZZZZZZZZ", "00045    ,8888888888888,  `                                 ,ZZZZZZZZZZZZZZZZZZZZZZZZ", "00046   d888888888888888,                                ,ZZZZZZZZZZZZZZZZZZZZZZZZZZZ", "00047   888888888888888888a,      _                    ,ZZZZZZZZZZZZZZZZZZZZ888888888", "00048   888888888888888888888ba,_d`                  ,ZZZZZZZZZZZZZZZZZ88888888888888", "00049   8888888888888888888888888888bbbaaa,,,______,ZZZZZZZZZZZZZZZ888888888888888888", "00050   88888888888888888888888888888888888888888ZZZZZZZZZZZZZZZ888888888888888888888", "00051   8888888888888888888888888888888888888888ZZZZZZZZZZZZZZ88888888888888888888888", "00052   888888888888888888888888888888888888888ZZZZZZZZZZZZZZ888888888888888888888888", "00053   8888888888888888888888888888888888888ZZZZZZZZZZZZZZ88888888888888888888888888", "00054   88888888888888888888888888888888888ZZZZZZZZZZZZZZ8888888888888888888888888888", "00055   8888888888888888888888888888888888ZZZZZZZZZZZZZZ88888888888888888 Normand  88", "00056   88888888888888888888888888888888ZZZZZZZZZZZZZZ8888888888888888888 Veilleux 88", "00057   8888888888888888888888888888888ZZZZZZZZZZZZZZ88888888888888888888888888888888", "00058   taken from: http://tinyurl.com/z2w9gu2"}
const SLEEP_TIME_BETWEEN_SERVER_CALLS_IN_MILLIS = 5000
const SLEEP_TIME_SHORT_IN_MILLIS = 100
const TIMESTAMP_WINDOW_IN_MINUTES = 8
const SEND_THRESHOLD = 2048
const MINE_ON_GENESIS = false
const MINE_HARDCODED = false
const OVERRIDE_DIFFICULTY = 0

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
	LisaIndex             int64
	mu                    sync.Mutex
}

type Slave struct {
	conn *rpc.Client
}

func main() {
	log.Debug("Entered main")
	master := &Master{}
	master.LisaIndex = -1
	master.Slaves = make([]Slave, 0)
	master.LastBlock = common.Block{}
	master.LastBlock.ParentId = ""
	master.NextSlave = 0
	rpc.Register(master)
	go listenForSlaves(master)

	for {
		log.Debug("Started main for loop")
		b := getNextBlock()
		if MINE_ON_GENESIS {
			b = getGenesisBlock()
		}
		if MINE_HARDCODED {
			b = getHardcodedBlock()
		}
		log.Debug("Waiting for lock in main for loop")
		master.mu.Lock()
		log.Debug("Acquired lock in main for loop")
		if b.ParentId != master.LastBlock.ParentId || (int64(time.Now().UnixNano())-int64(master.LastBlock.Timestamp))/NANOS_PER_MINUTE > TIMESTAMP_WINDOW_IN_MINUTES {
			master.LisaIndex += 1
			fmt.Println("Mining on new block")
			master.LastBlock = b
			master.LastBlock.Timestamp = uint64(time.Now().Add(time.Minute * time.Duration(TIMESTAMP_WINDOW_IN_MINUTES)).UnixNano())
			shaHash := sha256.Sum256([]byte(BLOCK_TEXTS[master.LisaIndex]))
			fmt.Println("New word: ", BLOCK_TEXTS[master.LisaIndex])
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
	if OVERRIDE_DIFFICULTY != 0 {
		data.Difficulty = OVERRIDE_DIFFICULTY
	}
	log.Debug("Returning in getNextBlock")

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

func getHardcodedBlock() common.Block {
	b := common.Block{}
	b.ParentId = "b8d8abb85732cad966bf0fe5b4d0e551d6917438d8ec251c1c74b49070e9cea8"
	b.Difficulty = 40
	b.Version = 0

	return b
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
			shaHash := sha256.Sum256([]byte(BLOCK_TEXTS[m.LisaIndex]))
			b.Root = hex.EncodeToString(shaHash[:])
			b.Difficulty = m.LastBlock.Difficulty
			b.Timestamp = m.LastBlock.Timestamp
			b.Nonces[0] = triple.Nonce1
			b.Nonces[1] = triple.Nonce2
			b.Nonces[2] = triple.Nonce3
			b.Version = m.LastBlock.Version
			commitBlock(m, b, BLOCK_TEXTS[m.LisaIndex])
		}
	}
	log.Debug("Releasing lock in SubmitAnswer")
	m.mu.Unlock()

	return nil
}
