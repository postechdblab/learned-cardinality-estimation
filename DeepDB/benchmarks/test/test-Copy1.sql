SELECT count(*) FROM link_type AS lt WHERE lt.link IN ('spoofs','spin off') ;
SELECT count(*) FROM link_type AS lt WHERE lt.link NOT IN ('spoofs','spin off') ;
SELECT count(*) FROM link_type AS lt WHERE lt.link LIKE 'follow%' ;
SELECT count(*) FROM link_type AS lt WHERE lt.link NOT LIKE 'follow%' ;
SELECT count(*) FROM link_type AS lt WHERE lt.link != 'remake of' ;
SELECT count(*) FROM link_type AS lt WHERE lt.link IS NOT NULL ;
SELECT count(*) FROM link_type AS lt WHERE lt.link IS NULL ;
SELECT count(*) FROM title AS t WHERE t.season_nr IS NULL ;
SELECT count(*) FROM title AS t WHERE t.season_nr IS NOT NULL ;