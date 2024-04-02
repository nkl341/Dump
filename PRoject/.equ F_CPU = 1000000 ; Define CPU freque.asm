.equ F_CPU = 1000000  ; Define CPU frequency as 1MHz

.equ PB0_PIN = 0       ; PB0 pin
.equ PC0_PIN = 0       ; PC0 pin

.equ HIGH_DUTY_CYCLE_DELAY = 660   ; 66% duty cycle delay in milliseconds
.equ LOW_DUTY_CYCLE_DELAY = 500    ; 50% duty cycle delay in milliseconds

; Register definitions
.equ DDRB_REG = 0x04   ; DDRB register address
.equ PORTB_REG = 0x05  ; PORTB register address
.equ PINB_REG = 0x03   ; PINB register address
.equ DDRC_REG = 0x07   ; DDRC register address
.equ PORTC_REG = 0x08  ; PORTC register address

.section .text
.global main

main:
    ; Set PB0 as input (logic level selection)
    ldi r16, ~(1 << PB0_PIN)
    out DDRB_REG, r16

    ; Enable pull-up resistor for PB0
    ldi r16, (1 << PB0_PIN)
    out PORTB_REG, r16

    ; Set PC0 as output
    ldi r16, (1 << PC0_PIN)
    out DDRC_REG, r16

loop:
    ; Check the input signal at PB0
    in r16, PINB_REG
    sbrs r16, PB0_PIN   ; Skip if PB0 is set
    rjmp low_duty_cycle

high_duty_cycle:
    ; PB0 is logic high, output 66% duty cycle
    sbr r16, (1 << PC0_PIN)  ; Set PC0 high
    out PORTC_REG, r16
    ldi r17, HIGH_DUTY_CYCLE_DELAY
    call delay_ms
    cbr r16, (1 << PC0_PIN)  ; Clear PC0
    out PORTC_REG, r16
    ldi r17, LOW_DUTY_CYCLE_DELAY
    call delay_ms
    rjmp loop

low_duty_cycle:
    ; PB0 is logic low, output 50% duty cycle
    sbr r16, (1 << PC0_PIN)  ; Set PC0 high
    out PORTC_REG, r16
    ldi r17, LOW_DUTY_CYCLE_DELAY
    call delay_ms
    cbr r16, (1 << PC0_PIN)  ; Clear PC0
    out PORTC_REG, r16
    ldi r17, LOW_DUTY_CYCLE_DELAY
    call delay_ms
    rjmp loop

delay_ms:
    ; Delay function
    ldi r18, 249
outer_loop:
    ldi r19, 32
inner_loop:
    dec r19
    brne inner_loop
    dec r18
    brne outer_loop
    ret

