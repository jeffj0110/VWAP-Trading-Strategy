from threading import Thread

#from robot_clean import TD_Robot_Loop

# Defines the global variables used to control the
# Bot from the UI and to also communicate the
# status of the bot back to the UI.
def init() :
    global Bot_Thread
    global Symbol
    global Bot_Status

    Symbol = ''
    Bot_Status = 'Not Started'

