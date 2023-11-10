package service;

import mapper.InterestRateRepository;
import org.hibernate.annotations.Source;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;

@Service
public class rateServer {
    @Resource
    private InterestRateRepository InterestRate;
}
