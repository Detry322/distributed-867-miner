package main

import (
	"bufio"
	"common"
	"encoding/binary"
	"encoding/hex"
	// "errors"
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
	address = "172.31.14.55:1337" //"18.187.0.66:1337"
	//address   = "107.20.178.226:1337" //"18.187.0.66:1337"
	path      = "/home/ubuntu/euphoric-gpu/miner/miner" // TODO: modify this to right path.
	maxChains = 2000
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
	Stdin        *bufio.Writer
	Stdout       *bufio.Reader
	MinerStarted bool
	HashChains   []common.HashChain
	configChan   chan common.HashConfig
	messageChan  chan string
}

/*
Parses message from miner.
*/
func (slave *Slave) ParseMessage(message string) {
	// message = strings.TrimFunc(message)
	//log.Debug("message from miner: " + message)
	reply := true
	if len(message) == 0 {
		return
	} else if string(message[0]) == "=" {
		// this is a debug message
		//log.Debug("debug message: " + message)
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
		timestamp := uint64(ts)
		// check if timestamp is good
		if timestamp != slave.Config.Block.Timestamp {
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
			hashchain := common.HashChain{
				Start:     uint64(x),
				End:       uint64(y),
				Length:    uint64(z),
				Timestamp: uint64(ts),
			}
			slave.mu.Lock()
			slave.HashChains = append(slave.HashChains, hashchain)
			if len(slave.HashChains) > maxChains {
				hashChains := slave.HashChains
				slave.HashChains = []common.HashChain{}
				slave.mu.Unlock()
				slave.Master.Call("Master.AddHashChains", hashChains, &reply)
			} else {
				slave.mu.Unlock()
			}
		} else if tokens[0] == "B" {
			collision := common.Collision{
				Nonce1:    uint64(x),
				Nonce2:    uint64(y),
				Nonce3:    uint64(z),
				Timestamp: uint64(ts),
			}
			slave.Master.Call("Master.SubmitAnswer", collision, &reply)
		}
	}
}

/*
This creates the message that sends a new hashconfig to the miner.
Ends in newline.
*/
func (slave *Slave) MakeHMessage() string {
	message := "H " + slave.Config.Block.ParentId +
		" " + slave.Config.Block.Root + " "
	message = common.AddHexDigits(message, slave.Config.Block.Difficulty, true)
	message += " "
	message = common.AddHexDigits(message, slave.Config.Block.Timestamp, true)
	message += " "
	// now just two hex bytes for version
	array := make([]byte, 2)
	binary.BigEndian.PutUint16(array, uint16(slave.Config.Block.Version))
	message += hex.EncodeToString(array[1:])
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
	log.Debug("about to loop through triples")
	for _, triple := range triples {
		message += " "
		message = common.AddHexDigits(message, triple.Chain1.Start, true)
		message += " "
		message = common.AddHexDigits(message, triple.Chain1.Length, false)
		message += " "
		message = common.AddHexDigits(message, triple.Chain2.Start, true)
		message += " "
		message = common.AddHexDigits(message, triple.Chain2.Length, false)
		message += " "
		message = common.AddHexDigits(message, triple.Chain3.Start, true)
		message += " "
		message = common.AddHexDigits(message, triple.Chain3.Length, false)
	}
	log.Debug("made part b message, returning")
	return message + "\n"
}

/*
RPC called by master
*/
func (slave *Slave) StartStepA(config common.HashConfig, reply *bool) (err error) {
	fmt.Println("tring to start part a")
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
	fmt.Println("actually starting part a")
	slave.Mode = A
	slave.Config = config
	hMessage := slave.MakeHMessage()
	io.WriteString(slave.Stdin, hMessage)
	slave.Stdin.Flush()
	aMessage := slave.MakeAMessage()
	log.Debug(hMessage)
	time.Sleep(2000 * time.Millisecond)
	io.WriteString(slave.Stdin, aMessage)
	slave.Stdin.Flush()
	log.Debug(aMessage)
	// send stuff to the miner
	return nil
}

// continuously checks if the process has output anything
func (slave *Slave) checkProcess() {
	var oldData []byte
	bufreader := (slave.Stdout)
	for {
		// read from stdoutpipe
		data, isPrefix, _ := bufreader.ReadLine()
		if len(data) == 0 {
			continue
		}
		fmt.Println(string(data))
		for _, d := range data {
			oldData = append(oldData, d)
		}
		if isPrefix {
			continue
			// oldData = append(oldData, data)
		} else {
			allData := oldData
			if len(allData) == 0 {
				continue
			}
			slave.messageChan <- string(allData)
			oldData = []byte{}
			// now send our message
			fmt.Println(string(allData))
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
	log.Debug("TRYING TO START PART B!")
	slave.mu.Lock()
	log.Debug("locked part b")
	defer slave.mu.Unlock()
	if config.Block.Timestamp < slave.Config.Block.Timestamp {
		// we ignore the request because timestamp is too small
		*reply = false
		log.Debug("timestamp too small")
		return nil
	} else if slave.Mode == B {
		log.Debug("already mode B")
		// send it to the channel.
		slave.configChan <- config
	}
	//empty hashchains, update config and send miner the triples.
	slave.HashChains = []common.HashChain{}
	log.Debug("ACTUALY STARTING PART B!!")
	if slave.Config.Block.Timestamp < config.Block.Timestamp {
		slave.Config = config
		io.WriteString(slave.Stdin, slave.MakeHMessage())
		slave.Stdin.Flush()
	} else if slave.Config.Block.Timestamp == config.Block.Timestamp {
		slave.Config = config
		//io.WriteString(slave.Stdin, slave.MakeHMessage())
		//slave.Stdin.Flush()
	}
	log.Debug("done with checking setting H again")
	io.WriteString(slave.Stdin, slave.MakeBMessage())
	log.Debug("wrote string")
	slave.Stdin.Flush()
	log.Debug("flushed")
	slave.Mode = B
	*reply = true
	log.Debug("returning before exiting part b")
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

func (slave *Slave) listen() {
	//go slave.checkProcess()
	reply := false
	for {
		select {
		case config := <-slave.configChan:
			if len(config.Triples) > 0 {
				slave.StartStepB(config, &reply)
			} else {
				slave.configChan <- config
			}
		case message := <-slave.messageChan:
			// new message to act on.
			slave.ParseMessage(message)
		default:
			<-time.NewTimer(time.Second / 4).C
		}
	}
}

func (slave *Slave) checkAlive() {
	slave.Cmd.Wait()
	log.Debug("Miner is done running!!!")
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
	slave.listen()
}

/*
This just starts the miner process
*/
func (slave *Slave) startMiner() {
	// start process, set slave.Cmd
	slave.Cmd = *exec.Command(path)
	// set stdin and stdout pipes
	stdinpipe, _ := slave.Cmd.StdinPipe()
	slave.Stdin = bufio.NewWriter(stdinpipe)
	stdoutpipe, _ := slave.Cmd.StdoutPipe()
	slave.Stdout = bufio.NewReader(stdoutpipe)
	(&slave.Cmd).Start()
	go slave.checkProcess()
	go slave.checkAlive()
	time.Sleep(3000 * time.Millisecond)
}

/*
Runs on startup
*/
func main() {
	slave := &Slave{
		configChan:  make(chan common.HashConfig, 100),
		messageChan: make(chan string, 1000),
	}
	rpc.Register(slave)
	slave.startMiner()
	go listenForMaster(slave) // runs forever, accepting RPCs.
	initiateConnection(slave)
}

func listenForMaster(slave *Slave) {
	// log.Debug("Listening for slaves")
	l, e := net.Listen("tcp", ":1336")
	if e != nil {
		log.Debug(e)
	}
	rpc.Accept(l)
}
