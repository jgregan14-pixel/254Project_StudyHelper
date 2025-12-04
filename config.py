'''
THis is the template below, if you wish to add more methods use the same format as below.
ensure you have a comma between each method.

    "NAMEOFTECHNIQUE": {
        "id": 1 (Unique number for method),
        "description": "WHAT IS IT?", 
        "work_time": Minutes * 60 (This is seconds),
        "break_time": Minutes * 60 (This is seconds),
        "cycles": 2 (How many cycles of work/break you want to do)
    }
'''

study_methods = {
    "pomodoro": {
        "id": 1,
        "description": "25 min study, 5 min break",
        "work_time": 25 * 60,
        "break_time": 5 * 60,
        "cycles": 4
    },
    "52-17": {
        "id": 2,
        "description": "52 min study, 17 min break",
        "work_time": 52 * 60,
        "break_time": 17 * 60,
        "cycles": 2
    },
    "custom": {
        "id": 3,
        "description": "Make your own study/break cycle",
        "work_time": None,   # handled by user input
        "break_time": None,  # handled by user input
        "cycles": None       # handled by user input
    },
}
