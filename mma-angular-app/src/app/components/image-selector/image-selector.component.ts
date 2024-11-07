import { Component } from '@angular/core';
import { ApiService } from '../../services/api.service';

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
          // Handle the response (e.g., display the comparison result)
        },
        (error) => {
          console.error('Error:', error);
          // Handle error (e.g., show an error message)
        }
      );
    } else {
      console.error('Both fighters must be selected.');
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