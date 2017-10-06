"""
This module validates the invalid JSON entries in the input file.
It also removes the unnecessary features from the input file.
"""


def format_input(input_file, preprocessed_input_file):

    with open(input_file) as data_file, open(preprocessed_input_file, 'w') as formatted_file:
        formatted_lines = []

        first_location = data_file.tell()
        last_line = data_file.readlines()[-1]
        data_file.seek(first_location)

        for line in data_file:
            line_list = line.split('", ')
            formatted_line_list = []

            for n_ll in line_list:
                i = n_ll.find(', "')
                s, t = '', ''

                if n_ll.startswith('"helpful"') or n_ll.startswith('"overall"'):
                    s = n_ll[:11] + '"' + n_ll[11:i] + '"'
                    t = n_ll[i + 2:] + '", '
                elif n_ll.startswith('"unixReview'):
                    # s = n_ll[:18] + '"' + n_ll[18:i] + '"'
                    # t = n_ll[i + 2:]
                    pass
                else:
                    s = n_ll + '"'

                formatted_line_list.append(s + ', ' + t)

            if line != last_line:
                formatted_lines.append(''.join(formatted_line_list)[:-4] + '}, \n')
            else:
                formatted_lines.append(''.join(formatted_line_list)[:-4] + '}')

        formatted_file.write('[')
        formatted_file.writelines(formatted_lines)
        formatted_file.write(']')
