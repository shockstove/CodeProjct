package com.example.demo.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.entity.InterestRate;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface InterestRateRepository extends BaseMapper<InterestRate> {
//    @Select("select rate from interest_rate where interest_type = #{type}")
//    List<Double> findAll();
}

