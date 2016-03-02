package main

import (
	"bufio"
	"common"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	log "github.com/Sirupsen/logrus"
	"io"
	"net"
	"net/rpc"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	A       = 1
	B       = 2
	address = "18.187.0.48:1337"
	path    = "~/miner" // TODO: modify this to right path.
)

func init() {
	// Only log the warning severity or above.
	log.SetLevel(log.DebugLevel)
}

type Slave struct {
	Config       common.HashConfig
	Id           uint64
	Mode         uint64
	mu           sync.Mutex
	Master       *rpc.Client
	Cmd          exec.Cmd
	Stdin        *io.Writer
	Stdout       *io.Reader
	MinerStarted bool
	HashChains   []common.HashChain
	configChan   chan *common.HashConfig
	messageChan  chan string
}

/*
Parses message from miner.
*/
func (slave *Slave) ParseMessage(message string) {
	message = strings.TrimFunc(message)
	if len(message) == 0 {
		return
	} else if message[0] == "=" {
		// this is a debug message
		log.Debug("debug message: " + message)
	} else {
		// non debug
		// A timestamp nonce nonce
		// B timestamp hash nonce
		tokens := strings.Fields(message)
		if len(tokens) != 5 {
			// invalid message
			log.Debug("Invalid message: not 5 tokens.")
			return
		}
		ts, e := strconv.Atoi(tokens[1])
		if e != nil {
			log.Debug("invalid timestamp")
			return
		}
		// check if timestamp is good
		if ts != slave.Config.Block.Timestamp {
			// we're done
			log.Debug("timestamp on miner msg is different")
			return
		}
		x, e1 := strconv.Atoi(tokens[2])
		y, e2 := strconv.Atoi(tokens[3])
		z, e3 := strconv.Atoi(tokens[2])
		if e1 != nil || e2 != nil || e3 != nil {
			log.Debug("invalid ints from miner")
			return
		}
		if tokens[0] == "A" {
			hashchain := common.HashChain {
				Start : x,
				End     :y,
				Length   :z,
				Timestamp :ts,
			}
		} else if tokens[0] == "B" {
			collision := common.CollisionB{X: uint64(x), Hash: uint64(yhash), Timestamp: uint64(ts)}
			slave.NewCollisionB(collision)
		}
	}
}

/*
This will send the miner a bytearray representing all the triples of triples he needs to check.
*/
func (slave *Slave) SendHashChainTripleList(triples []common.HashChainTriple, reply *bool) (err error) {
	*reply = true
	n, err := slave.Cmd.Stdin.Write(triples.convertToBytes())
	if err != nil {
		log.Debug("couldn't write everything for some reason.")
	}
	return nil
}

/*
This creates the message that sends a new hashconfig to the miner.
Ends in newline.
*/
func (slave *Slave) MakeHMessage() string {
	message := "H " + slave.Config.Block.ParentId +
		" " + slave.Config.Block.Root + " "
	common.AddHexDigits(message, slave.Config.Block.Diffulty, true)
	message += " "
	common.AddHexDigits(message, slave.Config.Block.Timestamp, true)
	message += " "
	// now just two hex bytes for version
	array := make([]byte, 4)
	binary.BigEndian.PutUint16(array, slave.Config.Block.Version)
	message += hex.EncodeToString(array)[2:]
	return message + "\n"
}

/*
This will create a message for switching to A mode.
*/
func (slave *Slave) MakeAMessage() string {
	return "A\n"
}

func (slave *Slave) MakeBMessage() string {
	triples := slave.Config.Triples
	message := "B"
	stream := []byte{}
	for _, triple := range triples {
		message += " "
		AddHexDigits(message, triple.Chain1.Start, true)
		message += " "
		AddHexDigits(message, triple.Chain1.Length, false)
		message += " "
		AddHexDigits(message, triple.Chain2.Start, true)
		message += " "
		AddHexDigits(message, triple.Chain2.Length, false)
		message += " "
		AddHexDigits(message, triple.Chain3.Start, true)
		message += " "
		AddHexDigits(message, triple.Chain3.Length, false)
	}
	return message + "\n"
}

/*
RPC called by master
*/
func (slave *Slave) StartStepA(config common.HashConfig, reply *bool) (err error) {
	slave.mu.Lock()
	defer slave.mu.Unlock()
	if config.Block.Timestamp <= slave.Config.Block.Timestamp {
		*reply = false
		return
	}
	if slave.Mode == B {
		// can't change the mode right now.
		slave.configChan <- config
		return
	}
	slave.Mode = A
	slave.Config = config
	io.WriteString(slave.Stdin, slave.MakeHMessage())
	io.WriteString(slave.Stdin, slave.MakeAMessage())
	// send stuff to the miner
	return nil
}

// continuously checks if the process has output anything
func (slave *Slave) checkProcess() {
	var oldData []byte
	for {
		// read from stdoutpipe
		data, isPrefix, err := slave.Stdout.ReadLine()
		if isPrefix {
			oldData = append(oldData, data)
		} else {
			allData = append(oldData, data)
			if len(allData == 0) {
				continue
			}
			oldData = make([]byte)
			// now send our message.
			slave.messageChan <- string(allData)
		}
	}
}

func (slave *Slave) NewHashChain(start, end, length, timestamp uint64) {
	if timestamp != slave.Config.Block.Timestamp {
		// basically we don't want it.
		return
	}
	hashchain := common.HashChain{
		Start:     start,
		End:       end,
		Length:    length,
		Timestamp: timestamp,
	}
	slave.HashChains = append(slave.HashChains, hashchain)

}

/*
This is used when we want this slave to transition from
*/
func (slave *Slave) StartStepB(config common.HashConfig, reply *bool) (err error) {
	slave.mu.Lock()
	defer slave.mu.Unlock()
	if config.Block.Timestamp < slave.Config.Block.Timestamp {
		// we ignore the request because timestamp is too small
		*reply = false
		return nil
	} else if slave.Mode == B {
		// send it to the channel.
		slave.configChan <- config
	}
	//empty hashchains, update config and send miner the triples.
	slave.HashChains = []common.HashChain{}
	if slave.Config.Block.Timestamp < config.Block.Timestamp {
		slave.Config = config
		io.WriteString(slave.Stdin, slave.MakeHMessage())
	}
	io.WriteString(slave.Stdin, slave.MakeBMessage())
	slave.Mode = B
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
	return "", nil
}

func (slave *Slave) listenToMiner() {
	for {

	}
}

func (slave *Slave) listen() {
	reply := false
	for {
		select {
			case config := <- slave.configChan {
				if len(config.Triples) > 0 {
					slave.StartStepB(config, &reply)
				} else {
					slave.configChan <- config
				}
			}
			case message <- slave.messageChan {
				// new message to act on.
				slave.ParseMessage(message)
			}
		}
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
	<-time.NewTimer(time.Second * 20).C
	err = slave.Master.Call("Master.Connect", ip, &reply)
	if err != nil {
		fmt.Println(err)
		return
	}
	slave.listen()
}

/*
This just starts the miner process
*/
func (slave *Slave) startMiner() {
	// start process, set slave.Cmd
	slave.Cmd = exec.Command(path)
	// set stdin and stdout pipes
	stdinpipe, err := slave.Cmd.StdinPipe()
	slave.Stdin = &bufio.NewWriter(stdinpipe)
	stdoutpipe, err := slave.Cmd.StdoutPipe()
	slave.Stdout = &bufio.NewReader(stdoutpipe)
	// send jack necessary info.

}

/*
Runs on startup
*/
func main() {
	slave := &Slave{
		configChan:  make(chan common.Config, 100),
		messageChan: make(chan string, 1000),
	}
	slave.startMiner()
	go listenForMaster(slave) // runs forever, accepting RPCs.
	initiateConnection(slave)
}

func listenForMaster(slave *Slave) {
	// log.Debug("Listening for slaves")
	l, e := net.Listen("tcp", ":1337")
	if e != nil {
		log.Debug("error on listen")
	}
	rpc.Accept(l)
}
