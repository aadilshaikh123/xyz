#include <stdio.h>
#include <limits.h>

char board[3][3];  // Game board

// Function to initialize the board
void initializeBoard() {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            board[i][j] = ' ';
}

// Function to display the board
void displayBoard() {
    printf("\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf(" %c ", board[i][j]);
            if (j < 2) printf("|");
        }
        printf("\n");
        if (i < 2) printf("---|---|---\n");
    }
    printf("\n");
}

// Check winner
int isWinner(char player) {
    for (int i = 0; i < 3; i++) {
        if (board[i][0] == player && board[i][1] == player && board[i][2] == player) return 1;
        if (board[0][i] == player && board[1][i] == player && board[2][i] == player) return 1;
    }
    if (board[0][0] == player && board[1][1] == player && board[2][2] == player) return 1;
    if (board[0][2] == player && board[1][1] == player && board[2][0] == player) return 1;
    return 0;
}

// Check draw
int isDraw() {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (board[i][j] == ' ') return 0;
    return 1;
}

// ---------- AI Technique (Minimax) ----------
int evaluateBoard() {
    if (isWinner('O')) return 10;
    if (isWinner('X')) return -10;
    return 0;
}

int minimax(int depth, int isMaximizing) {
    int score = evaluateBoard();
    if (score == 10 || score == -10) return score;
    if (isDraw()) return 0;

    int bestScore = isMaximizing ? INT_MIN : INT_MAX;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == ' ') {
                board[i][j] = isMaximizing ? 'O' : 'X';
                int moveScore = minimax(depth + 1, !isMaximizing);
                board[i][j] = ' ';
                if (isMaximizing)
                    bestScore = (moveScore > bestScore) ? moveScore : bestScore;
                else
                    bestScore = (moveScore < bestScore) ? moveScore : bestScore;
            }
        }
    }
    return bestScore;
}

void aiMoveMinimax() {
    int bestScore = INT_MIN;
    int moveRow = -1, moveCol = -1;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == ' ') {
                board[i][j] = 'O';
                int moveScore = minimax(0, 0);
                board[i][j] = ' ';
                if (moveScore > bestScore) {
                    bestScore = moveScore;
                    moveRow = i;
                    moveCol = j;
                }
            }
        }
    }
    board[moveRow][moveCol] = 'O';
}

// ---------- Non-AI Technique (posswin logic) ----------
int posswin(char player) {
    for (int i = 0; i < 3; i++) {
        // Rows
        if (board[i][0] == player && board[i][1] == player && board[i][2] == ' ') return i * 3 + 2;
        if (board[i][0] == player && board[i][2] == player && board[i][1] == ' ') return i * 3 + 1;
        if (board[i][1] == player && board[i][2] == player && board[i][0] == ' ') return i * 3 + 0;

        // Columns
        if (board[0][i] == player && board[1][i] == player && board[2][i] == ' ') return 6 + i;
        if (board[0][i] == player && board[2][i] == player && board[1][i] == ' ') return 3 + i;
        if (board[1][i] == player && board[2][i] == player && board[0][i] == ' ') return i;
    }

    // Diagonals
    if (board[0][0] == player && board[1][1] == player && board[2][2] == ' ') return 8;
    if (board[0][0] == player && board[2][2] == player && board[1][1] == ' ') return 4;
    if (board[1][1] == player && board[2][2] == player && board[0][0] == ' ') return 0;
    if (board[0][2] == player && board[1][1] == player && board[2][0] == ' ') return 6;
    if (board[0][2] == player && board[2][0] == player && board[1][1] == ' ') return 4;
    if (board[1][1] == player && board[2][0] == player && board[0][2] == ' ') return 2;

    return -1;
}

void aiMovePosswin() {
    int move;
    move = posswin('O');
    if (move != -1) {
        board[move / 3][move % 3] = 'O';
        return;
    }

    move = posswin('X');
    if (move != -1) {
        board[move / 3][move % 3] = 'O';
        return;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (board[i][j] == ' ') {
                board[i][j] = 'O';
                return;
            }
}

// ---------- Play Games ----------
void playGame(int useAI) {
    int row, col, player = 1;
    initializeBoard();

    while (1) {
        displayBoard();
        if (player == 1) {
            printf("Your turn! Enter row and column (1-3): ");
            scanf("%d %d", &row, &col);
            row--; col--;
            if (row < 0 || row >= 3 || col < 0 || col >= 3 || board[row][col] != ' ') {
                printf("Invalid move! Try again.\n");
                continue;
            }
            board[row][col] = 'X';
        } else {
            printf("AI is making a move...\n");
            if (useAI)
                aiMoveMinimax();
            else
                aiMovePosswin();
        }

        if (isWinner('X')) {
            displayBoard();
            printf("You win!\n");
            break;
        }
        if (isWinner('O')) {
            displayBoard();
            printf("AI wins!\n");
            break;
        }
        if (isDraw()) {
            displayBoard();
            printf("It's a draw!\n");
            break;
        }

        player = 3 - player;  // Switch turns
    }
}

// ---------- Main Menu ----------
int main() {
    int choice;
    printf("===== Tic-Tac-Toe Menu =====\n");
    printf("1. Play Tic-Tac-Toe (AI Technique - Minimax)\n");
    printf("2. Play Tic-Tac-Toe (Non-AI Technique - Posswin)\n");
    printf("Enter your choice: ");
    scanf("%d", &choice);

    if (choice == 1)
        playGame(1);
    else if (choice == 2)
        playGame(0);
    else
        printf("Invalid choice. Exiting.\n");

    return 0;
}
