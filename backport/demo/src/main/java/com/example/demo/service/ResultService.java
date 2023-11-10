package com.example.demo.service;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.demo.entity.Result;
import com.example.demo.mapper.ResultRepository;
import org.apache.ibatis.mapping.ResultMap;
import org.springframework.stereotype.Service;

@Service
public class ResultService extends ServiceImpl<ResultRepository, Result> {
}
