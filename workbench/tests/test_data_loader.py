from unittest import TestCase

import workbench.src.data_loader as data_loader


class Test(TestCase):
    def test__possession_parse(self):
        raw_xml = """<possession><value><comment>56</comment><event_incident_typefk>352</event_incident_typefk><elapsed>25</elapsed><subtype>possession</subtype><sortorder>1</sortorder><awaypos>44</awaypos><homepos>56</homepos><n>68</n><type>special</type><id>379029</id></value><value><comment>54</comment><elapsed_plus>1</elapsed_plus><event_incident_typefk>352</event_incident_typefk><elapsed>45</elapsed><subtype>possession</subtype><sortorder>4</sortorder><awaypos>46</awaypos><homepos>54</homepos><n>117</n><type>special</type><id>379251</id></value><value><comment>54</comment><event_incident_typefk>352</event_incident_typefk><elapsed>70</elapsed><subtype>possession</subtype><sortorder>0</sortorder><awaypos>46</awaypos><homepos>54</homepos><n>190</n><type>special</type><id>379443</id></value><value><comment>55</comment><elapsed_plus>5</elapsed_plus><event_incident_typefk>352</event_incident_typefk><elapsed>90</elapsed><subtype>possession</subtype><sortorder>1</sortorder><awaypos>45</awaypos><homepos>55</homepos><n>252</n><type>special</type><id>379575</id></value></possession>"""
        home_possession, away_possession = data_loader._possession_parse(raw_xml)

        print(f"home_possession:{home_possession}, away_possession:{away_possession}")

    def test__possession_parse_2(self):
        raw_xml = """<possession><value><comment>49</comment><event_incident_typefk>352</event_incident_typefk><elapsed>23</elapsed><subtype>possession</subtype><sortorder>2</sortorder><awaypos>51</awaypos><homepos>49</homepos><n>53</n><type>special</type><id>2745619</id></value><value><comment>46</comment><event_incident_typefk>352</event_incident_typefk><elapsed>45</elapsed><subtype>possession</subtype><sortorder>0</sortorder><awaypos>54</awaypos><homepos>46</homepos><n>107</n><type>special</type><id>2745937</id></value><value><comment>41</comment><event_incident_typefk>352</event_incident_typefk><elapsed>71</elapsed><subtype>possession</subtype><sortorder>0</sortorder><awaypos>59</awaypos><homepos>41</homepos><n>159</n><type>special</type><id>2746487</id></value><value><comment>47</comment><event_incident_typefk>352</event_incident_typefk><elapsed>82</elapsed><subtype>possession</subtype><sortorder>0</sortorder><awaypos>53</awaypos><homepos>47</homepos><n>189</n><type>special</type><id>2746629</id></value><value><comment>48</comment><elapsed_plus>3</elapsed_plus><event_incident_typefk>352</event_incident_typefk><elapsed>90</elapsed><subtype>possession</subtype><del>1</del><sortorder>0</sortorder><n>217</n><type>special</type><id>2746802</id></value><value><comment>47</comment><elapsed_plus>3</elapsed_plus><event_incident_typefk>352</event_incident_typefk><elapsed>90</elapsed><subtype>possession</subtype><sortorder>3</sortorder><awaypos>53</awaypos><homepos>47</homepos><n>225</n><type>special</type><id>2746835</id></value></possession>"""

        home_possession, away_possession = data_loader._possession_parse(raw_xml)
        print(f"home_possession:{home_possession}, away_possession:{away_possession}")

    def test__possession_parse_missing_str(self):
        for raw_xml in ["<NA>", None, ""]:
            home_possession, away_possession = data_loader._possession_parse(raw_xml)
            print(
                f"home_possession:{home_possession}, away_possession:{away_possession}"
            )
