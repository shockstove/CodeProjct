package com.example.demo.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.demo.entity.InterestRate;
import com.example.demo.entity.loanRate;
import com.example.demo.mapper.InterestRateRepository;
import com.example.demo.mapper.LoanRateRepository;
import com.example.demo.service.rateServer;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import javax.annotation.Resource;

    @Controller
    public class LoanRateController {

        @Resource
        private rateServer rateserver;
        @Resource
        private LoanRateRepository rateRepository;

        @GetMapping("/loan")
        @CrossOrigin(originPatterns = "*", methods = {RequestMethod.GET, RequestMethod.POST})
        @ResponseBody
        public String loanCal(@RequestParam("time") String time){
            QueryWrapper<loanRate> intQueWrapper=new QueryWrapper<>();
            double year= Double.parseDouble(time);
            if(year<=0.5)
            {
                intQueWrapper.eq("rate_type","六个月");
            }
            else if(year<=1)
            {
                intQueWrapper.eq("rate_type","一年");
            }
            else if(year<=3)
            {
                intQueWrapper.eq("rate_type","一至三年");
            }
            else if(year<=5)
            {
                intQueWrapper.eq("rate_type","三至五年");
            }
            else
            {
                intQueWrapper.eq("rate_type","五年");
            }
            loanRate intRate=rateRepository.selectOne(intQueWrapper);
            String result=Double.toString(intRate.getRate());
            return result;
        }

    }
