.equ inputPin = 2     ; Pin 2 is used as input
.equ outputPin = 3    ; Pin 3 is used as output

.equ DDRD_REG = 0x0A  ; DDRD register address
.equ PORTD_REG = 0x0B ; PORTD register address
.equ PIND_REG = 0x09  ; PIND register address

.section .text
.global main

main:
    ; Set inputPin as input
    ldi r16, ~(1 << inputPin)
    out DDRD_REG, r16

    ; Set outputPin as output
    ldi r16, (1 << outputPin)
    out DDRD_REG, r16

loop:
    ; Read the state of the input pin
    in r16, PIND_REG
    sbrs r16, inputPin  ; Skip if inputPin is clear
    rjmp input_high

input_low:
    ; Set the output pin LOW
    cbr r16, (1 << outputPin)
    out PORTD_REG, r16
    rjmp end_loop

input_high:
    ; Set the output pin HIGH
    sbr r16, (1 << outputPin)
    out PORTD_REG, r16

end_loop:
    ; Add delay if needed
    ; Implement delay function here

    ; Infinite loop
    rjmp loop
