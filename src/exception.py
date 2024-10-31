import sys


def error_message_details(error , error_details:str):

    _,_,exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_file

    error_message = 'Error occured in python script name [{0}] line number [{1}] error message[{2}]'.formate(
        file_name , exc_tb.tb_lineno , str(error)

    )
    return error_message


class CustomClass(Exception):
    def __init__(self , error_message ,error_details:sys):
        super.__init__(error_message)
        self.error_message = error_message_details(error_message , error_details=error_details)



    def __str__(self):
        return self.error_message