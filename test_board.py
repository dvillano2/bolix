import board_again
import torch

def test_board_height():
    assert board_again.board_height(5, 6) == 11
    assert board_again.board_height(2, 4) == 7 

def test_board_width():
    assert board_again.board_width(5, 6) == 19
    assert board_again.board_width(2, 4) == 9 

def test_batch_remove():
    side = 5
    depth = 6
    height = board_again.board_height(5, 6)
    width = board_again.board_width(5, 6)
    invalid_board = torch.ones - valid_board(5, 6)
