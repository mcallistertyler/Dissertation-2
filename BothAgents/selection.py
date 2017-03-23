import os
import initNetwork
import trainer
def selection():
	pswitch = False
	qswitch = False
	bswitch = False
	agentSelection = input("Choose an intelligent agent to use.\n 1 - Defensive Agent \n 2 - Offensive Agent \n")
	while agentSelection != "1" and agentSelection != "2":
		agentSelection = input("Choose an intelligent agent to use.\n 1 - Defensive Agent \n 2 - Offensive Agent \n")
	playMode = input("Choose a mode to play.\n 1 - Player against Agent \n 2 - Basic Computer against Agent \n 3 - Perfect Computer Against Agent \n")
	while  playMode != "1" and playMode != "2" and playMode != "3":
		playMode = input("Choose a mode to play.\n 1 - Player against Agent \n 2 - Basic Computer against Agent \n 3 - Perfect Computer Against Agent \n")		
	if playMode == "1":
		pswitch = True
	if playMode == "2":
		bswitch = True
	if playMode == "3":
		qswitch = True
	if agentSelection == "1":
		trainer.go(agentSelection, pswitch, qswitch, bswitch)
	if agentSelection == "2":
		trainer.go(agentSelection, pswitch, qswitch, bswitch)
selection()