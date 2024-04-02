.equ F_CPU = 1000000  ; Crystal frequency in Hz

; Define register aliases
.equ PORTB_REG = 0x05  ; PORTB register address
.equ DDRB_REG = 0x04   ; DDRB register address

; Define constants
.equ DELAY_MS = 10     ; Delay in milliseconds

; Define delay loop constants
.equ DELAY_CYCLES = (F_CPU / 1000) * DELAY_MS / 4

; Define bit masks
.equ PORTB_MASK = 0xAA  ; Initial value for PORTB

; Initialize stack pointer
LDI R16, low(RAMEND)
OUT SPL, R16
LDI R16, high(RAMEND)
OUT SPH, R16

; Set PORTB as output
LDI R16, 0xFF
OUT DDRB_REG, R16

; Set initial value of PORTB
LDI R16, PORTB_MASK
OUT PORTB_REG, R16

main:
    ; Toggle all bits of PORTB
    IN R16, PORTB_REG   ; Read current value of PORTB
    LDI R17, PORTB_MASK
    EOR R16, R17        ; Toggle all bits
    OUT PORTB_REG, R16  ; Write back to PORTB

    ; Delay
    LDI R17, LOW(DELAY_CYCLES)
    LDI R18, HIGH(DELAY_CYCLES)
delay_loop:
    DEC R17
    BRNE delay_loop
    DEC R18
    BRNE delay_loop

    ; Repeat
    RJMP main
