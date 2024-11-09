import { Component } from '@angular/core';
import { ApiService } from '../../services/api.service';

interface FighterStats {
  name: string;
  wins: number;
  losses: number;
  height: number;
  weight: number;
  reach: number;
  stance: string;
  age: number;
  slpm: number;
  strAcc: number;
  sapm: number;
  strDef: number;
  tdAvg: number;
  tdAcc: number;
  tdDef: number;
  subAvg: number;
}

@Component({
  selector: 'app-image-selector',
  templateUrl: './image-selector.component.html',
  styleUrls: ['./image-selector.component.css'],
})
export class ImageSelectorComponent {
  images = [
    { name: 'Nature', url: 'Screenshot 2023-04-24 214103.png' },
    { name: 'City', url: 'Screenshot 2023-06-07 200321.png' },
    { name: 'Mountains', url: 'Screenshot 2024-10-28 130023.png' },
  ];

  selectedImage = this.images[0].url;
  apiResponse: string = '';
  inputData: string = '';

  fighters: string[] = ['Islam Makhachev', 'Derrick Lewis', 'Brandon Moreno', 'Jon Jones'];
  selectedFighter1: string | undefined; // Declare this property
  selectedFighter2: string | undefined; // Declare this property
  fighter1Stats: FighterStats | null = null;
  fighter2Stats: FighterStats | null = null;

  constructor(private apiService: ApiService) {}

  onImageChange(event: Event): void {
    const target = event.target as HTMLSelectElement;
    this.selectedImage = target.value;
  }

  // Attempting to create a dual selection
  compareFighters() {
    if (this.selectedFighter1 && this.selectedFighter2) {
      const inputData = {
        fighter_1: this.selectedFighter1, // Make sure keys match what is expected in Lambda.
        fighter_2: this.selectedFighter2,
      };

      this.apiService.fetchData(inputData).subscribe(
        (response) => {
          console.log('API Response:', response);

          const parsedBody = JSON.parse(response.body);
          console.log('Parsed body:', parsedBody);

          if (parsedBody.fighter_1) {
            this.fighter1Stats = {
              name: parsedBody.fighter_1[0],
              wins: parsedBody.fighter_1[1],
              losses: parsedBody.fighter_1[2],
              height: parsedBody.fighter_1[3],
              weight: parsedBody.fighter_1[4],
              reach: parsedBody.fighter_1[5],
              stance: parsedBody.fighter_1[6],
              age: parsedBody.fighter_1[7],
              slpm: parsedBody.fighter_1[8],
              strAcc: parsedBody.fighter_1[9],
              sapm: parsedBody.fighter_1[10],
              strDef: parsedBody.fighter_1[11],
              tdAvg: parsedBody.fighter_1[12],
              tdAcc: parsedBody.fighter_1[13],
              tdDef: parsedBody.fighter_1[14],
              subAvg: parsedBody.fighter_1[15]
            };
          console.log('fighter1Stats:', this.fighter1Stats);
          }

          if (parsedBody.fighter_2) {
            this.fighter2Stats = {
              name: parsedBody.fighter_2[0],
              wins: parsedBody.fighter_2[1],
              losses: parsedBody.fighter_2[2],
              height: parsedBody.fighter_2[3],
              weight: parsedBody.fighter_2[4],
              reach: parsedBody.fighter_2[5],
              stance: parsedBody.fighter_2[6],
              age: parsedBody.fighter_2[7],
              slpm: parsedBody.fighter_2[8],
              strAcc: parsedBody.fighter_2[9],
              sapm: parsedBody.fighter_2[10],
              strDef: parsedBody.fighter_2[11],
              tdAvg: parsedBody.fighter_2[12],
              tdAcc: parsedBody.fighter_2[13],
              tdDef: parsedBody.fighter_2[14],
              subAvg: parsedBody.fighter_2[15]
            };
          console.log('fighter2Stats:', this.fighter2Stats);
          }
        },
        (error) => {
          console.error('Error:', error);
          this.apiResponse = 'Error occurred while fetching data.';
        }
      );
    } else {
      console.error('Both fighters must be selected.');
      this.apiResponse = 'Please select both fighters.';
    }
  }

  fetchApiData(inputData: string): void {
    this.apiService.fetchData(inputData).subscribe(
      (data) => {
        this.apiResponse = JSON.stringify(data); // Handle your API response here
      },
      (error) => {
        console.error('Error fetching data', error);
      }
    );
  }
  
}