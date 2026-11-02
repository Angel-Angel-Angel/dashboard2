-- REFERENCE TEMPLATE: actual CSV export query is unverified.

SELECT CAST(zip AS CHAR) AS zip,
       state_id, state_name, county_name, city,
       population, income_household_median, rent_median
FROM datamining.uszips
ORDER BY zip
