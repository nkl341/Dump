; Define register aliases
.equ R20 = 20
.equ R16 = 16
.equ R17 = 17
.equ R18 = 18

; Define port addresses
.equ PORTB = 0x25

; Define constants
.equ ADD_VALUE = 3
.equ LOOP_COUNT = 10

; Initialize stack pointer
LDI R16, high(RAMEND)
OUT SPH, R16
LDI R16, low(RAMEND)
OUT SPL, R16

; Set up PORTB for output
LDI R16, 0xFF     ; Set all bits of DDRB (PORTB direction register) to 1 for output
OUT DDRB, R16     ; Configure PORTB for output

; Clear R20
CLR R20

; Loop to add 3 to R20 ten times
LDI R17, LOOP_COUNT   ; Load loop counter

add_loop:
    ADD R20, ADD_VALUE  ; Add 3 to R20
    DEC R17             ; Decrement loop counter
    BRNE add_loop       ; Branch if not equal to zero to repeat the loop

; Send the sum to PORTB
OUT PORTB, R20

; End of program
END
