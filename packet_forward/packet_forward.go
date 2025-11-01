package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
)

func handleConnection(conn net.Conn, forward_host string) {
	fmt.Printf("Client connected: %v\n", conn.RemoteAddr())
	scanner := bufio.NewScanner(conn)
	for scanner.Scan() {
		fmt.Printf("Received Text\n%s\n", scanner.Text())
		conn, err := net.Dial("tcp", forward_host)
		if err != nil {
			fmt.Printf("Reciever Connection Error: %v\n", err)
			continue
		}
		// defer conn.Close()
		writer := bufio.NewWriter(conn)
		writer.Write(scanner.Bytes())
		if err := writer.Flush(); err != nil {
			fmt.Printf("Error Writing Bytes: %v\n", err)
		}
		//close writer and connection
		conn.Close()

	}
	if err := scanner.Err(); err != nil {
		fmt.Println("Sender Connection error:", err)
	}
	fmt.Println("Client disconnected.")
}

func forward_json_new_line_delimited(port int32, forward_host string) {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%v", port))
	if err != nil {
		panic(err)
	}
	defer listener.Close()
	fmt.Printf("Listening on  %s...\n", fmt.Sprintf(":%v", port))

	for {
		conn, err := listener.Accept()
		if err != nil {
			fmt.Println("Accept error:", err)
			continue
		}
		go handleConnection(conn, forward_host)
	}
}

func main() {
	forward_host := os.Args[1]
	forward_json_new_line_delimited(7450, forward_host)
}
