from util.args_help import fill_from_args
from tableqa.rci_system import RCISystem, TableQAOptions


if __name__ == "__main__":
    opts = TableQAOptions()
    fill_from_args(opts)
    rci = RCISystem(opts)
    header = ['Participant', 'Race', 'Date']
    rows = [['Michael', 'Runathon', 'June 10, 2020'],
            ['Mustafa', 'Runathon', 'Sept 3, 2020'],
            ['Alfio', 'Runathon', 'Jan 1, 2021'],]
    print(rci.get_answers(
        'Who won the race in June?',
        header, rows))

    # and separately for rows and columns
    print(rci.get_answer_columns('Who won the race in 2021?', header, rows))
    print(rci.get_answer_rows('Who won the race in 2021?', header, rows))
