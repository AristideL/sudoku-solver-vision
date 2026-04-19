class SudokuGrid:
    """Stores a 9x9 sudoku grid."""
    def __init__(self, grid: list[list[int]] | None = None):
        if grid is None:
            self.grid = [[0] * 9 for _ in range(9)]
        else:
            self.grid = grid
    
    def set_cell(self, row: int, col: int, value: int) -> None:
        """Set a cell value (0-9)."""
        if 0 <= row < 9 and 0 <= col < 9 and 0 <= value <= 9:
            self.grid[row][col] = value
    
    def get_cell(self, row: int, col: int) -> int:
        """Get a cell value."""
        if 0 <= row < 9 and 0 <= col < 9:
            return self.grid[row][col]
        return 0
    
    def __str__(self) -> str:
        """Return grid as formatted string."""
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)

def solve_sudoku(sudoku_grid: 'SudokuGrid') -> 'SudokuGrid':
    """Solve a sudoku puzzle using backtracking."""
    grid = [row[:] for row in sudoku_grid.grid]
    
    def is_valid(row: int, col: int, num: int) -> bool:
        # Check row
        if num in grid[row]:
            return False
        
        # Check column
        if num in [grid[i][col] for i in range(9)]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if grid[i][j] == num:
                    return False
        return True

    def backtrack() -> bool:
        for i in range(9):
            for j in range(9):
                if grid[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(i, j, num):
                            grid[i][j] = num
                            if backtrack():
                                return True
                            grid[i][j] = 0
                    return False
        return True

    backtrack()
    return SudokuGrid(grid)