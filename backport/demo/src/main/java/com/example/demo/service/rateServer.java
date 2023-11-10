package com.example.demo.service;

import com.example.demo.mapper.InterestRateRepository;
import com.example.demo.mapper.LoanRateRepository;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;

@Service
public class rateServer {
    @Resource
    private InterestRateRepository InterestRate;
    @Resource
    private LoanRateRepository LoanRate;
}
