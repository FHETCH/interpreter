from sympy import fft, ifft

FT_DIGITS_PRECISION = 15

def set_ft_decimal_precision(precision):
    global FT_DIGITS_PRECISION
    FT_DIGITS_PRECISION = precision

def forward_ft(data): 
    return fft(data, FT_DIGITS_PRECISION)

def inverse_ft(data): 
    return ifft(data, FT_DIGITS_PRECISION)
